/* tern_vec_dot.c — scalar reference ternary dot product.
 *
 * TQ{K}P vec_dot: multiply-free dot product of ternary-quantized weights
 * with Q8_K activation.  Compatible with sasori/tqkp.h block layout.
 *
 * Compile: gcc -shared -fPIC -O2 -o libtern_kernel.so tern_vec_dot.c
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>

#define QK_TQKP 256

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; e++; } while ((m & 0x400u) == 0);
            m &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Extract trit at logical position p from strided plane qs (tq2_0 layout). */
static inline int trit_at(const uint8_t *qs, int p) {
    int byte = (p / 128) * 32 + (p % 32);
    int bp   = (p / 32) % 4;
    return ((qs[byte] >> (bp * 2)) & 3) - 1;
}

/* Multiplication-free dot product: one weight row (TQ{K}P) with Q8_K activation.
 * Only multiplies are the K block scales — all else is add/subtract.
 */
float tern_vec_dot_row_kplane(
    const uint8_t  *row_data,  /* packed row data */
    int             in,         /* input dimension */
    int             group,      /* scale group size */
    int             K,          /* number of ternary planes */
    const uint8_t  *q8_act,     /* Q8_K quantized activation (qs + d + bsums) */
    float           act_d       /* Q8_K block scale */
) {
    const int qb = group / 4;       /* bytes per plane per block */
    const int nb = in / group;      /* number of blocks */
    const size_t bpb = (size_t)K * qb + (size_t)2 * K;  /* bytes per (row, block) */

    float acc = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t *blk     = row_data + (size_t)b * bpb;
        const uint8_t *q8_blk  = q8_act  + (size_t)b * QK_TQKP;

        /* Q8_K block sum for bias correction */
        int32_t q8_sum = 0;
        const uint8_t *bsums_ptr = q8_act + nb * QK_TQKP + (size_t)b * 4;
        __builtin_memcpy(&q8_sum, bsums_ptr, 4);

        for (int k = 0; k < K; k++) {
            const uint8_t *qs = blk + (size_t)k * qb;
            uint16_t dh_raw;
            __builtin_memcpy(&dh_raw, blk + (size_t)K * qb + (size_t)k * 2, 2);
            float scale = fp16_to_fp32(dh_raw) * act_d;

            int32_t sum = 0;
            int32_t act_sum = 0;

            for (int p = 0; p < group; p++) {
                int trit = trit_at(qs, p);
                int q8  = q8_blk[p];
                sum     += trit * q8;
                act_sum += q8;
            }

            /* Bias correction: ternary codes are {-1,0,+1}, but maddubs
               computes them as 0,1,2,3.  The ACTUAL dot is sum(trit_i * q8_i)
               = sum((u_i - 1) * q8_i) = sum(u_i * q8_i) - sum(q8_i).
               Since we already compute trit directly, no correction needed here.
            */
            acc += scale * (float)sum;
        }
    }

    return acc;
}

/* Full matrix-vector: all output rows, one Q8_K activation.
 * Writes results to dst[0..out-1].
 */
void tern_matvec_q8_K(
    const uint8_t  *packed_weight,  /* out * nb * bpb bytes */
    int             out,
    int             in,
    int             group,
    int             K,
    const uint8_t  *q8_act,
    float           act_d,
    float          *dst
) {
    const int nb = in / group;
    const size_t bpb = (size_t)K * (group / 4) + (size_t)2 * K;

    for (int r = 0; r < out; r++) {
        dst[r] = tern_vec_dot_row_kplane(
            packed_weight + (size_t)r * nb * bpb,
            in, group, K, q8_act, act_d
        );
    }
}

/* Simple dequantize: unpack ternary to float. */
void tern_dequantize_row_kplane(
    const uint8_t *data,
    int            out,
    int            in,
    int            group,
    int            K,
    float         *dst
) {
    const int nb = in / group;
    const size_t bpb = (size_t)K * (group / 4) + (size_t)2 * K;

    for (int r = 0; r < out; r++) {
        const uint8_t *row = data + (size_t)r * nb * bpb;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + (size_t)b * bpb;
            float *dptr = dst + (size_t)r * in + (size_t)b * group;

            for (int p = 0; p < group; p++) {
                float val = 0.0f;
                for (int k = 0; k < K; k++) {
                    uint16_t dh_raw;
                    __builtin_memcpy(&dh_raw, blk + (size_t)K * (group / 4) + (size_t)k * 2, 2);
                    int trit = trit_at(blk + (size_t)k * (group / 4), p);
                    val += fp16_to_fp32(dh_raw) * (float)trit;
                }
                dptr[p] = val;
            }
        }
    }
}
