#include "ggml-spin8.h"
#include "ggml-spin8-codec.h"

#include <cmath>

static double spin8_determinant(double matrix[8][8]) {
    double result = 1.0;
    for (int column = 0; column < 8; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 8; ++row) {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (matrix[pivot][column] == 0.0) {
            return 0.0;
        }
        if (pivot != column) {
            for (int j = 0; j < 8; ++j) {
                const double temp = matrix[column][j];
                matrix[column][j] = matrix[pivot][j];
                matrix[pivot][j] = temp;
            }
            result = -result;
        }
        const double diagonal = matrix[column][column];
        result *= diagonal;
        for (int row = column + 1; row < 8; ++row) {
            const double ratio = matrix[row][column] / diagonal;
            for (int j = column + 1; j < 8; ++j) {
                matrix[row][j] -= ratio * matrix[column][j];
            }
        }
    }
    return result;
}

int spin8_g128_validate_bank_f64(const double * bank, size_t bank_elements) {
    if (bank == nullptr || bank_elements != SPIN8_BANK_ELEMENTS) {
        return SPIN8_INVALID_BANK;
    }
    for (int frame = 0; frame < SPIN8_FRAMES; ++frame) {
        double matrix[8][8];
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                const double value = bank[frame * 64 + i * 8 + j];
                if (!std::isfinite(value)) {
                    return SPIN8_INVALID_BANK;
                }
                matrix[i][j] = value;
            }
        }
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                double inner = 0.0;
                for (int k = 0; k < 8; ++k) {
                    inner += matrix[i][k] * matrix[j][k];
                }
                if (std::abs(inner - (i == j ? 1.0 : 0.0)) > 1e-10) {
                    return SPIN8_INVALID_BANK;
                }
            }
        }
        if (std::abs(spin8_determinant(matrix) - 1.0) > 1e-10) {
            return SPIN8_INVALID_BANK;
        }
    }
    return SPIN8_OK;
}

int spin8_g128_validate_payload(const uint8_t * packed, size_t packed_bytes) {
    if (packed == nullptr || packed_bytes == 0 || packed_bytes % SPIN8_GROUP_BYTES != 0) {
        return SPIN8_INVALID_ARGUMENT;
    }
    const size_t groups = packed_bytes / SPIN8_GROUP_BYTES;
    for (size_t group = 0; group < groups; ++group) {
        const uint8_t * data = packed + group * SPIN8_GROUP_BYTES;
        const uint16_t scale = spin8_read_u16(data + 32);
        if (!spin8_scale_valid(scale)) {
            return SPIN8_INVALID_SCALE;
        }
        if ((scale & 0x8000) == 0) {
            for (int block = 0; block < 16; ++block) {
                if (spin8_read_u16(data + 2 * block) >= SPIN8_CODE_LIMIT) {
                    return SPIN8_INVALID_CODE;
                }
            }
        }
    }
    return SPIN8_OK;
}

int spin8_g128_validate_f64(const uint8_t * packed, size_t packed_bytes, const double * bank,
                           size_t bank_elements, int64_t rows, int64_t width) {
    if (!spin8_dimensions_valid(rows, width, 1, width, rows, sizeof(double))) {
        return SPIN8_INVALID_SHAPE;
    }
    const size_t groups = (size_t) rows * ((size_t) width / SPIN8_GROUP);
    if (packed_bytes != groups * SPIN8_GROUP_BYTES) {
        return SPIN8_INVALID_ARGUMENT;
    }
    const int status = spin8_g128_validate_payload(packed, packed_bytes);
    return status == SPIN8_OK ? spin8_g128_validate_bank_f64(bank, bank_elements) : status;
}

static void spin8_hadamard(double values[SPIN8_GROUP]) {
    for (int stride = 1; stride < SPIN8_GROUP; stride *= 2) {
        for (int block = 0; block < SPIN8_GROUP; block += 2 * stride) {
            for (int i = 0; i < stride; ++i) {
                const double a = values[block + i];
                const double b = values[block + stride + i];
                values[block + i] = a + b;
                values[block + stride + i] = a - b;
            }
        }
    }
    const double normalization = 1.0 / std::sqrt((double) SPIN8_GROUP);
    for (int i = 0; i < SPIN8_GROUP; ++i) {
        values[i] *= normalization;
    }
}

static double spin8_group_dot(const uint8_t * data, const double * bank, const double transformed[SPIN8_GROUP]) {
    const uint16_t scale_bits = spin8_read_u16(data + 32);
    const bool uniform = (scale_bits & 0x8000) != 0;
    const double scale = spin8_scale_magnitude(scale_bits);
    double result = 0;
    for (int block = 0; block < 16; ++block) {
        const int word = spin8_read_u16(data + 2 * block);
        if (uniform) {
            for (int i = 0; i < 8; ++i) {
                result += (((word >> (2 * i)) & 3) - 1.5) * transformed[block * 8 + i];
            }
            continue;
        }
        const bool identity = word >= SPIN8_IDENTITY_FIRST;
        int rank = identity ? spin8_identity_rank(word - SPIN8_IDENTITY_FIRST) : word % SPIN8_RADIX;
        const int frame = word / SPIN8_RADIX;
        for (int i = 0; i < 8; ++i) {
            const int trit = rank % 3 - 1;
            rank /= 3;
            if (trit == 0) {
                continue;
            }
            double rotated = transformed[block * 8 + i];
            if (!identity) {
                rotated = 0;
                for (int j = 0; j < 8; ++j) {
                    rotated += bank[frame * 64 + i * 8 + j] * transformed[block * 8 + j];
                }
            }
            result += trit * rotated;
        }
    }
    return scale * result;
}

template<typename T>
static double spin8_dot_row(const uint8_t * packed_row, const double * bank, const T * x, int64_t width) {
    double result = 0;
    for (int64_t group = 0; group < width / SPIN8_GROUP; ++group) {
        double transformed[SPIN8_GROUP];
        for (int i = 0; i < SPIN8_GROUP; ++i) {
            transformed[i] = x[group * SPIN8_GROUP + i];
        }
        spin8_hadamard(transformed);
        result += spin8_group_dot(packed_row + group * SPIN8_GROUP_BYTES, bank, transformed);
    }
    return result;
}

double spin8_g128_dot_f64(const uint8_t * packed_row, const double * bank, const float * x, int64_t width) {
    return spin8_dot_row(packed_row, bank, x, width);
}

void spin8_g128_decode_row_f32(const uint8_t * packed_row, const double * bank, float * output, int64_t width) {
    for (int64_t group = 0; group < width / SPIN8_GROUP; ++group) {
        const uint8_t * data = packed_row + group * SPIN8_GROUP_BYTES;
        const uint16_t scale_bits = spin8_read_u16(data + 32);
        const bool uniform = (scale_bits & 0x8000) != 0;
        const double scale = spin8_scale_magnitude(scale_bits);
        double decoded[SPIN8_GROUP];
        for (int block = 0; block < 16; ++block) {
            const int word = spin8_read_u16(data + 2 * block);
            const bool identity = word >= SPIN8_IDENTITY_FIRST;
            const int rank = uniform ? 0 : (identity ? spin8_identity_rank(word - SPIN8_IDENTITY_FIRST) : word % SPIN8_RADIX);
            const int frame = word / SPIN8_RADIX;
            for (int i = 0; i < 8; ++i) {
                double value = 0;
                if (uniform) {
                    value = ((word >> (2 * i)) & 3) - 1.5;
                } else if (identity) {
                    value = spin8_trit(rank, i);
                } else {
                    int digits = rank;
                    for (int j = 0; j < 8; ++j) {
                        value += (digits % 3 - 1) * bank[frame * 64 + j * 8 + i];
                        digits /= 3;
                    }
                }
                decoded[block * 8 + i] = scale * value;
            }
        }
        spin8_hadamard(decoded);
        for (int i = 0; i < SPIN8_GROUP; ++i) {
            output[group * SPIN8_GROUP + i] = (float) decoded[i];
        }
    }
}

int spin8_g128_matmul_cpu_f64(const uint8_t * packed, size_t packed_bytes,
                             const double * bank, const double * x, size_t x_elements,
                             double * output, size_t output_elements,
                             int64_t rows, int64_t width, int64_t tokens,
                             int64_t x_stride, int64_t output_stride) {
    if (!spin8_dimensions_valid(rows, width, tokens, x_stride, output_stride, sizeof(double))) {
        return SPIN8_INVALID_SHAPE;
    }
    const size_t x_span = (size_t) (tokens - 1) * x_stride + width;
    const size_t output_span = (size_t) (tokens - 1) * output_stride + rows;
    if (x == nullptr || output == nullptr || x_elements < x_span || output_elements < output_span) {
        return SPIN8_INVALID_ARGUMENT;
    }
    const int status = spin8_g128_validate_f64(packed, packed_bytes, bank, SPIN8_BANK_ELEMENTS, rows, width);
    if (status != SPIN8_OK) {
        return status;
    }
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t i = 0; i < width; ++i) {
            if (!std::isfinite(x[token * x_stride + i])) {
                return SPIN8_NONFINITE_INPUT;
            }
        }
    }
    const size_t row_bytes = (size_t) (width / SPIN8_GROUP) * SPIN8_GROUP_BYTES;
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t row = 0; row < rows; ++row) {
            const double result = spin8_dot_row(packed + row * row_bytes, bank, x + token * x_stride, width);
            if (!std::isfinite(result)) {
                return SPIN8_NONFINITE_OUTPUT;
            }
            output[token * output_stride + row] = result;
        }
    }
    return SPIN8_OK;
}

int spin8_g128_identity_rank(int index) {
    return index < 0 || index >= SPIN8_CODE_LIMIT - SPIN8_IDENTITY_FIRST ? -1 : spin8_identity_rank(index);
}

int spin8_g128_scale_value(uint16_t bits, double * magnitude) {
    if (magnitude == nullptr) {
        return SPIN8_INVALID_ARGUMENT;
    }
    if (!spin8_scale_valid(bits)) {
        return SPIN8_INVALID_SCALE;
    }
    *magnitude = spin8_scale_magnitude(bits);
    return SPIN8_OK;
}

const char * spin8_status_string(int status) {
    switch (status) {
        case SPIN8_OK:               return "success";
        case SPIN8_INVALID_ARGUMENT: return "invalid pointer or buffer capacity";
        case SPIN8_INVALID_SHAPE:    return "invalid or overflowing tensor dimensions";
        case SPIN8_INVALID_SCALE:    return "scale must be finite and nonzero";
        case SPIN8_INVALID_CODE:     return "reserved ternary codeword";
        case SPIN8_INVALID_BANK:     return "bank must contain nine SO(8) matrices";
        case SPIN8_NONFINITE_INPUT:  return "input activation is nonfinite";
        case SPIN8_NONFINITE_OUTPUT: return "output overflowed or became nonfinite";
        default:                    return "unknown status";
    }
}
