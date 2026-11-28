#include "ggml-so8t.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace {

__device__ __forceinline__ int so8t_swz_a(int row, int chunk) { return chunk ^ (row & 3); }
__device__ __forceinline__ int so8t_swz_b(int row, int chunk) { return chunk ^ (row & 7); }

__device__ __forceinline__ uint32_t so8t_mask_r(int a) {
    return (((a < 4) ? 0x53399500u : 0x4d278be1u) >> ((a & 3) * 8)) & 0xffu;
}
__device__ __forceinline__ uint32_t so8t_mask_rconj(int a) {
    return (((a < 4) ? 0xa4c268feu : 0x3298540eu) >> ((a & 3) * 8)) & 0xffu;
}
__device__ __forceinline__ uint32_t so8t_mask_l(int a) {
    return (((a < 4) ? 0xa5c36900u : 0x3399550fu) >> ((a & 3) * 8)) & 0xffu;
}

__device__ __forceinline__ float decode_e2m1_nibble(uint8_t code) {
    static constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = kMag[code & 0x07u];
    return ((code & 0x08u) != 0u) ? -v : v;
}

__global__ void kernel_dequantize_so8t_nvfp4_to_f16(
    const block_so8t_oct_nvfp4* __restrict__ blocks,
    __half* __restrict__ out_f16,
    int num_blocks) {
    const int blk_idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (blk_idx >= num_blocks) return;

    const block_so8t_oct_nvfp4 blk = blocks[blk_idx];
    const float scale = __half2float(__ushort_as_half(blk.scale_f16));
    __half* dst = out_f16 + static_cast<size_t>(blk_idx) * 16;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint8_t byte = blk.qs[i];
        dst[i * 2 + 0] = __float2half(decode_e2m1_nibble(byte & 0x0Fu) * scale);
        dst[i * 2 + 1] = __float2half(decode_e2m1_nibble((byte >> 4) & 0x0Fu) * scale);
    }
}

// On-the-fly exact Frobenius-orthogonal projection of plain `.safetensors` dense matrix W[N, K]
// (where standard linear layer computes Y = X * W^T, so 8x8 block B[a, k] = W[og*8 + k, ig*8 + a])
// onto Cayley-Dickson octonion weights W_oct[og, ig, 8]:
//   w_b^* = (1/8) * sum_{a=0..7} s_R(a, a^b) * B[a, a^b]
__global__ void kernel_project_safetensors_f16_to_oct(
    const __half* __restrict__ w_dense_nk,  // [N, K] row-major (standard HuggingFace safetensors layout)
    __half* __restrict__ w_oct,             // [out_groups, in_groups, 8]
    int in_groups,
    int out_groups) {
    const int oct_idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int total_octs = out_groups * in_groups;
    if (oct_idx >= total_octs) return;

    const int og = oct_idx / in_groups;
    const int ig = oct_idx % in_groups;
    const int K = in_groups * 8;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const uint32_t neg_mask = so8t_mask_r(a);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            // Standard safetensors Linear weight is [out_features, in_features] = [og*8 + k, ig*8 + a]
            const float val = __half2float(w_dense_nk[static_cast<size_t>(og * 8 + k) * K + (ig * 8 + a)]);
            const float s = ((neg_mask >> k) & 1u) ? -1.0f : 1.0f;
            acc[a ^ k] += s * val;
        }
    }

    __half* dst = w_oct + static_cast<size_t>(oct_idx) * 8;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        dst[b] = __float2half(acc[b] * 0.125f);
    }
}

__device__ __forceinline__ void so8t_expand_triality_shfl(
    __half* __restrict__ dst0, __half* __restrict__ dst1, __half* __restrict__ dst2,
    const __half* __restrict__ w_oct_ptr, int a) {
    const uint32_t w_lane = static_cast<uint32_t>(__half_as_ushort(w_oct_ptr[a]));
    uint32_t v[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = __shfl_xor_sync(0xffffffffu, w_lane, k, 8) & 0xffffu;
    auto pack = [&](uint32_t mask) -> uint4 {
        auto s = [&](int k) -> uint32_t { return (v[k] ^ (((mask >> k) & 1u) << 15)) & 0xffffu; };
        uint4 p;
        p.x = s(0) | (s(1) << 16); p.y = s(2) | (s(3) << 16);
        p.z = s(4) | (s(5) << 16); p.w = s(6) | (s(7) << 16);
        return p;
    };
    *reinterpret_cast<uint4*>(dst0) = pack(so8t_mask_r(a));
    *reinterpret_cast<uint4*>(dst1) = pack(so8t_mask_rconj(a));
    *reinterpret_cast<uint4*>(dst2) = pack(so8t_mask_l(a));
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void so8t_cp_async_16b(void* smem_ptr, const void* gmem_ptr) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem_ptr) : "memory");
}
__device__ __forceinline__ void so8t_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void so8t_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}
__device__ __forceinline__ void so8t_ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void so8t_ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void so8t_mma_m16n8k16_f16(
    uint32_t& d0, uint32_t& d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}
#endif

__global__ void __launch_bounds__(256, 2)
kernel_ggml_so8t_triality_triple_tc_gemm(
    const __half* __restrict__ X,
    const __half* __restrict__ W_oct,
    float* __restrict__ Y_v,
    float* __restrict__ Y_sp,
    float* __restrict__ Y_sm,
    int M, int in_groups, int out_groups) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int kBM = 64, kBN = 64, kBK = 32, kStages = 2, kSwizzleM = 8, kPromoteTiles = 8;
    __shared__ alignas(16) __half smem_a[kStages][kBM][kBK];
    __shared__ alignas(16) __half smem_b[kStages][3][kBK][kBN];

    const int K = in_groups * 8, N = out_groups * 8;
    const int grid_m = M / kBM, grid_n = N / kBN;
    const int linear_id = static_cast<int>(blockIdx.y) * grid_n + static_cast<int>(blockIdx.x);
    const int group_size = kSwizzleM * grid_n;
    const int group_id = linear_id / group_size;
    const int first_tile_m = group_id * kSwizzleM;
    const int active_gm = (grid_m - first_tile_m < kSwizzleM) ? (grid_m - first_tile_m) : kSwizzleM;
    const int tile_m = first_tile_m + (linear_id % active_gm);
    const int tile_n = (linear_id % group_size) / active_gm;

    const int block_row = tile_m * kBM, block_col = tile_n * kBN, block_out_g = tile_n * 8;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5, lane = tid & 31;
    const int warp_row = (warp_id / 4) * 32, warp_col = (warp_id % 4) * 16;

    const int a_row = tid / 4, a_chunk = tid % 4, a_col = a_chunk * 8;
    const int a_swz_col = so8t_swz_a(a_row, a_chunk) * 8;

    const int b_oct = tid >> 3, b_row_a = tid & 7, b_og = b_oct >> 2, b_ig = b_oct & 3;
    const int b_smem_row = b_ig * 8 + b_row_a;
    const int b_swz_col = so8t_swz_b(b_smem_row, b_og) * 8;

    uint32_t acc_f16[3][2][2][2];
    float acc_f32[3][2][2][4];
    #pragma unroll
    for (int t = 0; t < 3; ++t)
        #pragma unroll
        for (int wi = 0; wi < 2; ++wi)
            #pragma unroll
            for (int wj = 0; wj < 2; ++wj) {
                acc_f16[t][wi][wj][0] = 0u; acc_f16[t][wi][wj][1] = 0u;
                acc_f32[t][wi][wj][0] = 0.0f; acc_f32[t][wi][wj][1] = 0.0f;
                acc_f32[t][wi][wj][2] = 0.0f; acc_f32[t][wi][wj][3] = 0.0f;
            }

    const int num_k_tiles = K / kBK;
    auto issue_a = [&](int kt, int st) {
        so8t_cp_async_16b(&smem_a[st][a_row][a_swz_col], X + static_cast<size_t>(block_row + a_row) * K + (kt * kBK + a_col));
        so8t_cp_async_commit();
    };
    auto expand_b = [&](int kt, int st) {
        const size_t w_off = (static_cast<size_t>(block_out_g + b_og) * in_groups + (kt * 4 + b_ig)) * 8;
        so8t_expand_triality_shfl(&smem_b[st][0][b_smem_row][b_swz_col],
                                  &smem_b[st][1][b_smem_row][b_swz_col],
                                  &smem_b[st][2][b_smem_row][b_swz_col], W_oct + w_off, b_row_a);
    };

    issue_a(0, 0);
    expand_b(0, 0);
    so8t_cp_async_wait<0>();
    __syncthreads();

    const int ld_a_row_off = lane & 15, ld_a_col_half = (lane >> 4) & 1;
    const int ld_b_k_off = lane & 15, ld_b_n_half = (lane >> 4) & 1;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int cur = k_tile & 1, nxt = cur ^ 1, next_tile = k_tile + 1;
        if (next_tile < num_k_tiles) issue_a(next_tile, nxt);

        #pragma unroll
        for (int k_sub = 0; k_sub < 2; ++k_sub) {
            uint32_t reg_a[2][4];
            #pragma unroll
            for (int wi = 0; wi < 2; ++wi) {
                const int r = warp_row + wi * 16 + ld_a_row_off;
                so8t_ldmatrix_x4(reg_a[wi][0], reg_a[wi][1], reg_a[wi][2], reg_a[wi][3],
                                 &smem_a[cur][r][so8t_swz_a(r, k_sub * 2 + ld_a_col_half) * 8]);
            }
            const int b_r = k_sub * 16 + ld_b_k_off;
            const int b_swz_c = so8t_swz_b(b_r, (warp_col / 8) + ld_b_n_half) * 8;
            #pragma unroll
            for (int t = 0; t < 3; ++t) {
                uint32_t reg_b[2][2];
                so8t_ldmatrix_x4_trans(reg_b[0][0], reg_b[0][1], reg_b[1][0], reg_b[1][1], &smem_b[cur][t][b_r][b_swz_c]);
                #pragma unroll
                for (int wi = 0; wi < 2; ++wi)
                    #pragma unroll
                    for (int wj = 0; wj < 2; ++wj)
                        so8t_mma_m16n8k16_f16(acc_f16[t][wi][wj][0], acc_f16[t][wi][wj][1],
                                              reg_a[wi][0], reg_a[wi][1], reg_a[wi][2], reg_a[wi][3],
                                              reg_b[wj][0], reg_b[wj][1],
                                              acc_f16[t][wi][wj][0], acc_f16[t][wi][wj][1]);
            }
        }

        if (next_tile < num_k_tiles) expand_b(next_tile, nxt);

        if (((k_tile + 1) % kPromoteTiles) == 0 || (k_tile + 1) == num_k_tiles) {
            #pragma unroll
            for (int t = 0; t < 3; ++t)
                #pragma unroll
                for (int wi = 0; wi < 2; ++wi)
                    #pragma unroll
                    for (int wj = 0; wj < 2; ++wj) {
                        const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&acc_f16[t][wi][wj][0]));
                        const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&acc_f16[t][wi][wj][1]));
                        acc_f32[t][wi][wj][0] += f0.x; acc_f32[t][wi][wj][1] += f0.y;
                        acc_f32[t][wi][wj][2] += f1.x; acc_f32[t][wi][wj][3] += f1.y;
                        acc_f16[t][wi][wj][0] = 0u; acc_f16[t][wi][wj][1] = 0u;
                    }
        }

        if (next_tile < num_k_tiles) { so8t_cp_async_wait<0>(); __syncthreads(); }
    }

    float* outs[3] = {Y_v, Y_sp, Y_sm};
    const int grp = lane >> 2, col2 = (lane & 3) * 2;
    #pragma unroll
    for (int t = 0; t < 3; ++t) {
        #pragma unroll
        for (int wi = 0; wi < 2; ++wi) {
            const int r0 = block_row + warp_row + wi * 16 + grp;
            #pragma unroll
            for (int wj = 0; wj < 2; ++wj) {
                const int c = block_col + warp_col + wj * 8 + col2;
                *reinterpret_cast<float2*>(outs[t] + static_cast<size_t>(r0) * N + c) =
                    make_float2(acc_f32[t][wi][wj][0], acc_f32[t][wi][wj][1]);
                *reinterpret_cast<float2*>(outs[t] + static_cast<size_t>(r0 + 8) * N + c) =
                    make_float2(acc_f32[t][wi][wj][2], acc_f32[t][wi][wj][3]);
            }
        }
    }
#else
    (void)X; (void)W_oct; (void)Y_v; (void)Y_sp; (void)Y_sm; (void)M; (void)in_groups; (void)out_groups;
#endif
}

}  // namespace

bool ggml_so8t_load_safetensors_f16_tensor(
    const std::string& path,
    const std::string& tensor_name,
    ggml_so8t_safetensors_tensor& out_tensor) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    uint64_t hdr_len = 0;
    ifs.read(reinterpret_cast<char*>(&hdr_len), sizeof(hdr_len));
    if (!ifs || hdr_len == 0 || hdr_len > (1ULL << 26)) return false;

    std::string header(static_cast<size_t>(hdr_len), '\0');
    ifs.read(header.data(), static_cast<std::streamsize>(hdr_len));
    if (!ifs) return false;

    // Locate `"<tensor_name>": { ... "dtype": "F16" ... "shape": [N, K] ... "data_offsets": [start, end] }`
    const std::string needle = "\"" + tensor_name + "\"";
    const size_t key_pos = header.find(needle);
    if (key_pos == std::string::npos) return false;
    const size_t obj_start = header.find('{', key_pos + needle.size());
    const size_t obj_end = header.find('}', obj_start);
    if (obj_start == std::string::npos || obj_end == std::string::npos) return false;
    const std::string obj = header.substr(obj_start, obj_end - obj_start + 1);

    if (obj.find("\"F16\"") == std::string::npos) return false;

    std::smatch m_shape, m_off;
    static const std::regex re_shape(R"(\"shape\"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\])");
    static const std::regex re_off(R"(\"data_offsets\"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\])");
    if (!std::regex_search(obj, m_shape, re_shape) || !std::regex_search(obj, m_off, re_off)) {
        return false;
    }

    const int64_t d0 = std::stoll(m_shape[1].str());
    const int64_t d1 = std::stoll(m_shape[2].str());
    const uint64_t off_begin = std::stoull(m_off[1].str());
    const uint64_t off_end = std::stoull(m_off[2].str());
    const uint64_t nbytes = off_end - off_begin;
    if (nbytes != static_cast<uint64_t>(d0) * static_cast<uint64_t>(d1) * 2ULL) return false;

    out_tensor.name = tensor_name;
    out_tensor.shape = {d0, d1};
    out_tensor.data_f16.resize(static_cast<size_t>(nbytes / 2ULL));
    ifs.seekg(static_cast<std::streamoff>(8ULL + hdr_len + off_begin), std::ios::beg);
    ifs.read(reinterpret_cast<char*>(out_tensor.data_f16.data()), static_cast<std::streamsize>(nbytes));
    return static_cast<bool>(ifs);
}

void ggml_so8t_dequantize_nvfp4_to_oct_f16_cuda(
    const block_so8t_oct_nvfp4* d_blocks,
    __half* d_w_oct_f16,
    int num_octonions,
    cudaStream_t stream) {
    const int num_blocks = num_octonions / 2;  // 2 octonions (16 elems) per block
    const int threads = 256;
    const int grid = (num_blocks + threads - 1) / threads;
    kernel_dequantize_so8t_nvfp4_to_f16<<<grid, threads, 0, stream>>>(d_blocks, d_w_oct_f16, num_blocks);
}

void ggml_so8t_project_safetensors_f16_to_oct_cuda(
    const __half* d_w_dense_nk,
    __half* d_w_oct,
    int in_groups,
    int out_groups,
    cudaStream_t stream) {
    const int total_octs = out_groups * in_groups;
    const int threads = 256;
    const int grid = (total_octs + threads - 1) / threads;
    kernel_project_safetensors_f16_to_oct<<<grid, threads, 0, stream>>>(d_w_dense_nk, d_w_oct, in_groups, out_groups);
}

void ggml_so8t_triality_triple_tc_gemm_cuda(
    const __half* d_x,
    const __half* d_w_oct,
    float* d_y_v,
    float* d_y_sp,
    float* d_y_sm,
    int m,
    int in_groups,
    int out_groups,
    cudaStream_t stream) {
    const int n = out_groups * 8;
    const dim3 grid(static_cast<unsigned int>(n / 64), static_cast<unsigned int>(m / 64), 1u);
    kernel_ggml_so8t_triality_triple_tc_gemm<<<grid, 256, 0, stream>>>(
        d_x, d_w_oct, d_y_v, d_y_sp, d_y_sm, m, in_groups, out_groups);
}

void ggml_so8t_safetensors_triple_infer_cuda(
    const __half* d_x,
    const __half* d_w_dense_nk,
    __half* d_w_oct_scratch,
    float* d_y_v,
    float* d_y_sp,
    float* d_y_sm,
    int m,
    int in_groups,
    int out_groups,
    cudaStream_t stream) {
    ggml_so8t_project_safetensors_f16_to_oct_cuda(d_w_dense_nk, d_w_oct_scratch, in_groups, out_groups, stream);
    ggml_so8t_triality_triple_tc_gemm_cuda(d_x, d_w_oct_scratch, d_y_v, d_y_sp, d_y_sm, m, in_groups, out_groups, stream);
}
