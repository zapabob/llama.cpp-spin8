#ifdef SPIN8_STANDALONE_CUDA
#include <cuda_runtime.h>
#else
#include "spin8.cuh"
#endif
#include "../ggml-spin8-codec.h"

struct spin8_input_layout {
    int64_t ne1;
    int64_t ne2;
    size_t stride1;
    size_t stride2;
    size_t stride3;
};

static __device__ __forceinline__ float spin8_component(int word, bool uniform, int coordinate, const float * bank) {
    if (uniform) {
        return (float) ((word >> (2 * coordinate)) & 3) - 1.5f;
    }
    if (word >= SPIN8_CODE_LIMIT) {
        return __int_as_float(0x7fc00000);
    }
    if (word >= SPIN8_IDENTITY_FIRST) {
        return (float) spin8_trit(spin8_identity_rank(word - SPIN8_IDENTITY_FIRST), coordinate);
    }
    const int frame = word / SPIN8_RADIX;
    int rank = word % SPIN8_RADIX;
    float value = 0.0f;
    for (int i = 0; i < 8; ++i) {
        value += (float) (rank % 3 - 1) * bank[frame * 64 + i * 8 + coordinate];
        rank /= 3;
    }
    return value;
}

static __device__ __forceinline__ float spin8_hadamard_f32(float value, float workspace[2][SPIN8_GROUP]) {
    const int lane = threadIdx.x;
    int current = 0;
    workspace[current][lane] = value;
    __syncthreads();
    for (int stride = 1; stride < SPIN8_GROUP; stride *= 2) {
        const float a = workspace[current][lane];
        const float b = workspace[current][lane ^ stride];
        workspace[1 - current][lane] = (lane & stride) != 0 ? b - a : a + b;
        __syncthreads();
        current = 1 - current;
    }
    return workspace[current][lane] * 0.08838834764831844055f;
}

static __global__ void spin8_mul_mat_f32(const uint8_t * packed, const double * bank_f64,
                                        const float * x, float * output, int64_t rows, int64_t width,
                                        int64_t tokens, spin8_input_layout layout, int64_t output_stride) {
#ifndef SPIN8_STANDALONE_CUDA
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
#endif
    __shared__ float bank[SPIN8_BANK_ELEMENTS];
    __shared__ float workspace[2][SPIN8_GROUP];
    const int lane = threadIdx.x;
    for (int i = lane; i < SPIN8_BANK_ELEMENTS; i += SPIN8_GROUP) {
        bank[i] = (float) bank_f64[i];
    }
    __syncthreads();
    const int64_t groups = width / SPIN8_GROUP;
    for (int64_t pair = blockIdx.x; pair < rows * tokens; pair += gridDim.x) {
        const int64_t row = pair % rows;
        const int64_t token = pair / rows;
        const size_t x_offset = (token % layout.ne1) * layout.stride1 +
            ((token / layout.ne1) % layout.ne2) * layout.stride2 +
            (token / layout.ne1 / layout.ne2) * layout.stride3;
        float sum = 0.0f;
        for (int64_t group = 0; group < groups; ++group) {
            const float transformed = spin8_hadamard_f32(x[x_offset + group * SPIN8_GROUP + lane], workspace);
            const uint8_t * data = packed + ((size_t) row * groups + group) * SPIN8_GROUP_BYTES;
            const uint16_t scale_bits = spin8_read_u16(data + 32);
            const int word = spin8_read_u16(data + 2 * (lane / 8));
            const float scale = spin8_scale_valid(scale_bits) ? spin8_scale_magnitude(scale_bits) : __int_as_float(0x7fc00000);
            const float decoded = spin8_component(word, (scale_bits & 0x8000) != 0, lane % 8, bank);
            sum += transformed * (scale * decoded);
        }
        workspace[0][lane] = sum;
        __syncthreads();
        for (int stride = SPIN8_GROUP / 2; stride > 0; stride /= 2) {
            if (lane < stride) {
                workspace[0][lane] += workspace[0][lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0) {
            output[token * output_stride + row] = workspace[0][0];
        }
        __syncthreads();
    }
}

static __global__ void spin8_get_rows_f32(const uint8_t * packed, const double * bank_f64,
                                         const int32_t * ids, float * output, int64_t rows,
                                         int64_t width, int64_t tokens) {
#ifndef SPIN8_STANDALONE_CUDA
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
#endif
    __shared__ float bank[SPIN8_BANK_ELEMENTS];
    __shared__ float workspace[2][SPIN8_GROUP];
    const int lane = threadIdx.x;
    for (int i = lane; i < SPIN8_BANK_ELEMENTS; i += SPIN8_GROUP) {
        bank[i] = (float) bank_f64[i];
    }
    __syncthreads();
    const int64_t groups = width / SPIN8_GROUP;
    for (int64_t pair = blockIdx.x; pair < tokens * groups; pair += gridDim.x) {
        const int64_t token = pair / groups;
        const int64_t group = pair % groups;
        const int64_t row = ids[token];
        if (row < 0 || row >= rows) {
            output[token * width + group * SPIN8_GROUP + lane] = __int_as_float(0x7fc00000);
            continue;
        }
        const uint8_t * data = packed + ((size_t) row * groups + group) * SPIN8_GROUP_BYTES;
        const uint16_t scale_bits = spin8_read_u16(data + 32);
        const int word = spin8_read_u16(data + 2 * (lane / 8));
        const float scale = spin8_scale_valid(scale_bits) ? spin8_scale_magnitude(scale_bits) : __int_as_float(0x7fc00000);
        const float decoded = spin8_component(word, (scale_bits & 0x8000) != 0, lane % 8, bank);
        output[token * width + group * SPIN8_GROUP + lane] = spin8_hadamard_f32(scale * decoded, workspace);
    }
}

static cudaError_t spin8_launch_matmul(const uint8_t * packed, const double * bank, const float * x,
                                     float * output, int64_t rows, int64_t width, int64_t tokens,
                                     spin8_input_layout layout, int64_t output_stride, cudaStream_t stream) {
    const int64_t work = rows * tokens;
    const unsigned blocks = (unsigned) (work < 65535 ? work : 65535);
#ifdef SPIN8_STANDALONE_CUDA
    spin8_mul_mat_f32<<<blocks, SPIN8_GROUP, 0, stream>>>(packed, bank, x, output, rows, width, tokens, layout, output_stride);
    return cudaGetLastError();
#else
    const ggml_cuda_kernel_launch_params params(blocks, SPIN8_GROUP, 0, stream);
    ggml_cuda_kernel_launch(spin8_mul_mat_f32, params, packed, bank, x, output, rows, width, tokens, layout, output_stride);
    return cudaSuccess;
#endif
}

extern "C" cudaError_t spin8_g128_cuda_matmul_f32(const uint8_t * packed, const double * bank, const float * x,
                                                float * output, int64_t rows, int64_t width, int64_t tokens,
                                                int64_t x_stride, int64_t output_stride, cudaStream_t stream) {
    if (packed == nullptr || bank == nullptr || x == nullptr || output == nullptr ||
        !spin8_dimensions_valid(rows, width, tokens, x_stride, output_stride, sizeof(float))) {
        return cudaErrorInvalidValue;
    }
    const spin8_input_layout layout = {tokens, 1, (size_t) x_stride, 0, 0};
    return spin8_launch_matmul(packed, bank, x, output, rows, width, tokens, layout, output_stride, stream);
}

extern "C" cudaError_t spin8_g128_cuda_get_rows_f32(const uint8_t * packed, const double * bank, const int32_t * ids,
                                                  float * output, int64_t rows, int64_t width,
                                                  int64_t tokens, cudaStream_t stream) {
    if (packed == nullptr || bank == nullptr || ids == nullptr || output == nullptr || tokens <= 0 ||
        !spin8_dimensions_valid(rows, width, 1, width, rows, sizeof(float)) ||
        (uint64_t) tokens > SIZE_MAX / sizeof(float) / (uint64_t) width ||
        tokens > INT64_MAX / (width / SPIN8_GROUP)) {
        return cudaErrorInvalidValue;
    }
    const int64_t work = tokens * (width / SPIN8_GROUP);
    const unsigned blocks = (unsigned) (work < 65535 ? work : 65535);
#ifdef SPIN8_STANDALONE_CUDA
    spin8_get_rows_f32<<<blocks, SPIN8_GROUP, 0, stream>>>(packed, bank, ids, output, rows, width, tokens);
    return cudaGetLastError();
#else
    const ggml_cuda_kernel_launch_params params(blocks, SPIN8_GROUP, 0, stream);
    ggml_cuda_kernel_launch(spin8_get_rows_f32, params, packed, bank, ids, output, rows, width, tokens);
    return cudaSuccess;
#endif
}

#ifndef SPIN8_STANDALONE_CUDA
static void spin8_assert_tensors(const ggml_tensor * weights, const ggml_tensor * bank, const ggml_tensor * dst) {
    GGML_ASSERT(weights->type == GGML_TYPE_SPIN8_G128 && weights->ne[2] == 1 && weights->ne[3] == 1);
    GGML_ASSERT(bank->type == GGML_TYPE_F64 && bank->ne[0] == 8 && bank->ne[1] == 8 && bank->ne[2] == 9 && bank->ne[3] == 1);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(weights) && ggml_is_contiguous(bank));
}

void ggml_cuda_op_mul_mat_spin8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * bank = dst->src[2];
    spin8_assert_tensors(weights, bank, dst);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float));
    GGML_ASSERT(x->ne[0] == weights->ne[0] && dst->ne[0] == weights->ne[1]);
    const int64_t tokens = ggml_nrows(x);
    GGML_ASSERT(ggml_nrows(dst) == tokens);
    GGML_ASSERT(spin8_dimensions_valid(weights->ne[1], weights->ne[0], tokens, weights->ne[0], weights->ne[1], sizeof(float)));
    size_t span = x->ne[0];
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        GGML_ASSERT(x->nb[i] % sizeof(float) == 0 && x->ne[i] > 0);
        const size_t stride = x->nb[i] / sizeof(float);
        GGML_ASSERT(stride == 0 || (size_t) (x->ne[i] - 1) <= (SIZE_MAX / sizeof(float) - span) / stride);
        span += (x->ne[i] - 1) * stride;
    }
    const spin8_input_layout layout = {x->ne[1], x->ne[2], x->nb[1] / sizeof(float), x->nb[2] / sizeof(float), x->nb[3] / sizeof(float)};
    CUDA_CHECK(spin8_launch_matmul((const uint8_t *) weights->data, (const double *) bank->data,
                                  (const float *) x->data, (float *) dst->data, weights->ne[1], weights->ne[0],
                                  tokens, layout, weights->ne[1], ctx.stream()));
}

void ggml_cuda_op_get_rows_spin8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    const ggml_tensor * bank = dst->src[2];
    spin8_assert_tensors(weights, bank, dst);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    GGML_ASSERT(dst->ne[0] == weights->ne[0] && ggml_nrows(dst) == ggml_nelements(ids));
    CUDA_CHECK(spin8_g128_cuda_get_rows_f32((const uint8_t *) weights->data, (const double *) bank->data,
                                          (const int32_t *) ids->data, (float *) dst->data,
                                          weights->ne[1], weights->ne[0], ggml_nelements(ids), ctx.stream()));
}
#endif
