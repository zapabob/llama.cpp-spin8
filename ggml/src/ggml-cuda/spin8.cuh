#ifndef GGML_CUDA_SPIN8_CUH
#define GGML_CUDA_SPIN8_CUH
#include "common.cuh"

void ggml_cuda_op_mul_mat_spin8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_get_rows_spin8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#endif
