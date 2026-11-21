#ifndef SPIN8_RUNTIME_H
#define SPIN8_RUNTIME_H

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum spin8_status {
    SPIN8_OK = 0,
    SPIN8_INVALID_ARGUMENT = 1,
    SPIN8_INVALID_SHAPE = 2,
    SPIN8_INVALID_SCALE = 3,
    SPIN8_INVALID_CODE = 4,
    SPIN8_INVALID_BANK = 5,
    SPIN8_NONFINITE_INPUT = 6,
    SPIN8_NONFINITE_OUTPUT = 7,
};

GGML_API const char * spin8_status_string(int status);

GGML_API int spin8_g128_validate_payload(const uint8_t * packed, size_t packed_bytes);
GGML_API int spin8_g128_validate_bank_f64(const double * bank, size_t bank_elements);

GGML_API int spin8_g128_validate_f64(const uint8_t * packed, size_t packed_bytes,
                           const double * bank, size_t bank_elements,
                           int64_t rows, int64_t width);

// Row kernels require a validated payload, bank, width, and finite input.
GGML_API double spin8_g128_dot_f64(const uint8_t * packed_row, const double * bank, const float * x, int64_t width);
GGML_API void spin8_g128_decode_row_f32(const uint8_t * packed_row, const double * bank, float * output, int64_t width);

// Strides and capacities are in elements. Output must not alias input or packed data.
GGML_API int spin8_g128_matmul_cpu_f64(const uint8_t * packed, size_t packed_bytes,
                             const double * bank, const double * x, size_t x_elements,
                             double * output, size_t output_elements,
                             int64_t rows, int64_t width, int64_t tokens,
                             int64_t x_stride, int64_t output_stride);

// Helpers for independent exhaustive tests of the serialized contract.
GGML_API int spin8_g128_identity_rank(int index);
GGML_API int spin8_g128_scale_value(uint16_t bits, double * magnitude);

#ifdef __cplusplus
}
#endif

#endif
