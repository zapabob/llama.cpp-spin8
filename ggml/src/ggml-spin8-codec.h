#ifndef SPIN8_CODEC_H
#define SPIN8_CODEC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC__
#define SPIN8_INLINE __host__ __device__ __forceinline__
#else
#define SPIN8_INLINE inline
#endif

enum {
    SPIN8_GROUP = 128,
    SPIN8_GROUP_BYTES = 34,
    SPIN8_FRAMES = 9,
    SPIN8_BANK_ELEMENTS = 576,
    SPIN8_RADIX = 6561,
    SPIN8_IDENTITY_FIRST = 59049,
    SPIN8_CODE_LIMIT = 65498,
};

SPIN8_INLINE uint16_t spin8_read_u16(const uint8_t * data) {
    return (uint16_t) (data[0] | ((uint16_t) data[1] << 8));
}

SPIN8_INLINE bool spin8_scale_valid(uint16_t bits) {
    return (bits & 0x7fff) != 0 && (bits & 0x7c00) != 0x7c00;
}

SPIN8_INLINE int spin8_pow3(int exponent) {
    int result = 1;
    for (int i = 0; i < exponent; ++i) {
        result *= 3;
    }
    return result;
}

SPIN8_INLINE int spin8_support_completions(int remaining, int support) {
    if (support == 0) {
        return 1;
    }
    if (support == 1) {
        return 2 * remaining;
    }
    if (support == 2) {
        return 2 * remaining * (remaining - 1);
    }
    return 0;
}

// Unrank the increasing ternary-rank list with support two excluded.
SPIN8_INLINE int spin8_identity_rank(int index) {
    int rank = 0;
    int support = 0;
    for (int position = 7; position >= 0; --position) {
        const int place = spin8_pow3(position);
        for (int digit = 0; digit < 3; ++digit) {
            const int next_support = support + (digit != 1);
            const int count = place - spin8_support_completions(position, 2 - next_support);
            if (index < count) {
                rank += digit * place;
                support = next_support;
                break;
            }
            index -= count;
        }
    }
    return rank;
}

SPIN8_INLINE int spin8_trit(int rank, int coordinate) {
    return (rank / spin8_pow3(coordinate)) % 3 - 1;
}

// All finite binary16 magnitudes are exact in binary32 and binary64.
SPIN8_INLINE float spin8_scale_magnitude(uint16_t bits) {
    const unsigned exponent = (bits >> 10) & 31;
    const unsigned mantissa = bits & 1023;
    float value = exponent == 0 ? (float) mantissa : (float) (1024 + mantissa);
    int shift = exponent == 0 ? -24 : (int) exponent - 25;
    while (shift > 0) {
        value *= 2.0f;
        --shift;
    }
    while (shift < 0) {
        value *= 0.5f;
        ++shift;
    }
    return value;
}

SPIN8_INLINE bool spin8_dimensions_valid(int64_t rows, int64_t width, int64_t tokens,
                                        int64_t x_stride, int64_t output_stride, size_t element_size) {
    if (rows <= 0 || width <= 0 || tokens <= 0 || width % SPIN8_GROUP != 0 ||
        x_stride < width || output_stride < rows || element_size == 0) {
        return false;
    }
    const uint64_t groups = (uint64_t) width / SPIN8_GROUP;
    if ((uint64_t) rows > SIZE_MAX / SPIN8_GROUP_BYTES / groups || rows > INT64_MAX / tokens) {
        return false;
    }
    const uint64_t limit = SIZE_MAX / element_size;
    if ((uint64_t) width > limit || (uint64_t) rows > limit) {
        return false;
    }
    return (uint64_t) (tokens - 1) <= (limit - width) / x_stride &&
           (uint64_t) (tokens - 1) <= (limit - rows) / output_stride;
}

#undef SPIN8_INLINE
#endif
