#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#else
struct __half;
typedef struct CUstream_st* cudaStream_t;
#endif

// ============================================================================
// GGML SO(8)T / Spin(8) Octonion Quantized Tensor Types & Safetensors Loader
// ============================================================================
// Reserved GGML type IDs for llama.cpp-spin8 GGUF v3 compatibility:
//   - GGML_TYPE_SO8T_OCT_F16  (108): 16B per 8x8 octonion block (8x / 24x Triality compression)
//   - GGML_TYPE_SO8T_OCT_NVFP4 (109): 10B per 2-octonion (16-elem) E2M1 + FP16 scale block (25.6x / 76.8x)

constexpr uint32_t GGML_TYPE_SO8T_OCT_F16   = 108U;
constexpr uint32_t GGML_TYPE_SO8T_OCT_NVFP4 = 109U;

#pragma pack(push, 1)
struct block_so8t_oct_f16 {
    uint16_t w_f16[8];  // 8 FP16 octonion components (16 bytes)
};

struct block_so8t_oct_nvfp4 {
    uint16_t scale_f16; // Shared FP16 block scale for 16 elements (2 octonions)
    uint8_t  qs[8];     // 16 packed 4-bit E2M1 codes (lo nibble = even, hi nibble = odd)
};
#pragma pack(pop)

static_assert(sizeof(block_so8t_oct_f16) == 16, "block_so8t_oct_f16 must be 16 bytes");
static_assert(sizeof(block_so8t_oct_nvfp4) == 10, "block_so8t_oct_nvfp4 must be 10 bytes");

struct ggml_so8t_safetensors_tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<uint16_t> data_f16;  // Raw IEEE-754 FP16 payload loaded directly from .safetensors
};

/// Zero-conversion loader: reads a standard HuggingFace `.safetensors` F16 tensor directly
/// (8-byte LE header length + JSON header + raw data slice) without any offline preprocessing.
bool ggml_so8t_load_safetensors_f16_tensor(
    const std::string& path,
    const std::string& tensor_name,
    ggml_so8t_safetensors_tensor& out_tensor);

/// Dequantize GGML_TYPE_SO8T_OCT_NVFP4 blocks directly on GPU into FP16 octonion weights
/// `d_w_oct[out_groups, in_groups, 8]` ready for Tensor Core Triality execution.
void ggml_so8t_dequantize_nvfp4_to_oct_f16_cuda(
    const block_so8t_oct_nvfp4* d_blocks,
    __half* d_w_oct_f16,
    int num_octonions,
    cudaStream_t stream = nullptr);

/// On-the-fly exact Frobenius-orthogonal projection of ANY plain `.safetensors` FP16 dense matrix
/// `d_w_dense[N, K]` (row-major, N = 8*out_groups, K = 8*in_groups) onto the 8D Cayley-Dickson
/// octonion subspace `d_w_oct[out_groups, in_groups, 8]`:
///   w_b^* = (1/8) * sum_{a=0..7} s_R(a, a^b) * B[a, a^b]
/// Requires ZERO offline retraining, ZERO SVD, and ZERO format conversion.
void ggml_so8t_project_safetensors_f16_to_oct_cuda(
    const __half* d_w_dense_nk,
    __half* d_w_oct,
    int in_groups,
    int out_groups,
    cudaStream_t stream = nullptr);

/// Spin(8) Triality Triple-Inference Fused Tensor Core GEMM (Direct PTX ldmatrix.x4 + mma.m16n8k16):
/// Simultaneously computes all three Cartan Triality outputs from ONE weight read (1/24 dense):
///   Y_v  (V_8)  = X * W       = X * R(w)
///   Y_sp (S_8+) = X * conj(W) = X * R(conj(w))
///   Y_sm (S_8-) = W * X       = X * L(w)
void ggml_so8t_triality_triple_tc_gemm_cuda(
    const __half* d_x,
    const __half* d_w_oct,
    float* d_y_v,
    float* d_y_sp,
    float* d_y_sm,
    int m,
    int in_groups,
    int out_groups,
    cudaStream_t stream = nullptr);

/// 1-call end-to-end plain `.safetensors` -> Spin(8) Triality Triple Inference (V_8, S_8+, S_8-):
/// Projects `d_w_dense_nk[N, K]` on-the-fly into `d_w_oct_scratch` and runs fused Triple-Inference TC GEMM.
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
    cudaStream_t stream = nullptr);
