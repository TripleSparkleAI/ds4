/* DeepSeek V4.1 Flash — CUDA port of the Metal primitives in metal/dsv41.metal + ds4_metal.m.
 *
 * <claudes_code_comments>
 * ** Function List **
 * dsv41_bf16_dev / dsv41_pow2_ceil_dev        - BF16 round-to-nearest-even; next power of two (Metal twins)
 * dsv41_warp_allsum_f32 / dsv41_warp_allmax_f32 - xor-butterfly reductions, result in EVERY lane (Metal simd_*)
 * dsv41_bf16_linear_kernel                    - in-place BF16 rounding of a float buffer
 * dsv41_rope_kernel                           - unit-magnitude RoPE on the last 64 dims, 32 lanes = 32 pairs
 * dsv41_quantize_kernel                       - BF16 / FP8-E8M0 / FP4-E8M0 / FP4-E4M3 fake-quant, per 32|16 block
 * dsv41_engram_add_kernel                     - gated Engram add: cos-sim gate over 4 heads x key rows
 * dsv41_pool2_kernel                          - softmax-weighted pair pooling of compressed KV
 * dsv41_candidate_blocks_kernel / _filter     - 8-wide block max + causal filter for the two-tier indexer
 * dsv41_carry_copy_kernel                     - compact prefill carry pack/unpack (bf16 halves or 32-bit mask)
 * dsv41_gather_kv_kernel                      - out[r] = source[ids[r]] for 512-wide rows
 * ds4_gpu_dsv41_*                             - the public wrappers (validation mirrors ds4_metal.m)
 * ds4_gpu_dsv41_attention_output_batch        - Q8 out_a -> BF16-round low -> Q8 out_b (impl hook in ds4_cuda.cu)
 * ds4_gpu_hc_rms_scale_project_f16_tensor     - mode-0 semantics: rms_norm_plain_rows + f16 matmul
 * stubs                                       - tensor ops (Metal MPP) off, TP off, seed_experts no-op
 *
 * ** Technical Review **
 * - 1:1 port of the Metal math. Every kernel keeps Metal's launch shape (threadgroup -> block, lane ->
 *   threadIdx.x & 31) so the reduction ORDER inside a warp matches Metal's simdgroup where it matters.
 * - Metal's simd_sum/simd_max broadcast to all lanes; ds4_cuda.cu's warp_sum_f32 is shfl_down (lane 0
 *   only), so this file carries its own xor-butterfly all-lane reductions.
 * - Launches go on cuda_decode_stream() so they capture into the decode CUDA graph like the rest.
 * - The packed (Metal tensor-op) indexer path is reported unavailable: ds4.c then takes the unpacked
 *   path (indexer_scores_batch -> the existing V4 indexer kernel, top-k -> the existing V4 top-k).
 * - hc_rms_scale_project uses the fused wrapper's documented mode-0 definition (rms-norm rows, then the
 *   f16 projection); the caller's flat_norm scratch is [rows][DS4_N_HC*DS4_N_EMBD], so it fits.
 * - Verification: golden = antirez's Metal build on the M5 with the same GGUF (greedy token identity),
 *   plus the release quality batch. Transcendentals (exp/cos/sin) are not bit-portable across GPUs, so
 *   small late drifts are expected and are judged on top-1 agreement (see plans/PLAN_SPARKPORT_*.md).
 * </claudes_code_comments>
 */

#ifndef DS4_V41_TYPES_DEFINED
#define DS4_V41_TYPES_DEFINED
/* V4.1 activation/cache formats. Buffers are float-addressable but the
 * rounded values follow the released BF16/FP8/FP4 inference graph. */
typedef enum {
    DS4_V41_BF16 = 0,
    DS4_V41_FP8_E8M0 = 1,
    DS4_V41_FP4_E8M0 = 2,
    DS4_V41_FP4_E4M3 = 3,
} ds4_v41_activation_format;
enum { DS4_V41_CARRY_BF16, DS4_V41_CARRY_MASK, DS4_V41_CARRY_F32 };
#endif /* DS4_V41_TYPES_DEFINED */

/* ---- device helpers ---- */
__device__ static inline float dsv41_bf16_dev(float x) {
    unsigned int bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}
__device__ static inline float dsv41_pow2_ceil_dev(float x) {
    const unsigned int bits = __float_as_uint(x);
    return __uint_as_float((bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0u));
}
__device__ static inline float dsv41_warp_allsum_f32(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ static inline float dsv41_warp_allmax_f32(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* ---- kernels (metal/dsv41.metal twins) ---- */
__global__ static void dsv41_bf16_linear_kernel(unsigned int *x, unsigned long long count) {
    const unsigned long long first = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 4ull;
    if (first + 4ull <= count) {
        uint4 b = *((uint4 *)(x + first));
        unsigned int *p = (unsigned int *)&b;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned int bits = p[k];
            if ((bits & 0x7f800000u) != 0x7f800000u) bits += 0x7fffu + ((bits >> 16u) & 1u);
            p[k] = bits & 0xffff0000u;
        }
        *((uint4 *)(x + first)) = b;
    } else {
        for (unsigned long long i = first; i < count; i++) {
            unsigned int bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u) bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct dsv41_rope_args { unsigned int width, heads, rows, start, inverse, stride; float frequencies[32]; };

__global__ static void dsv41_rope_kernel(dsv41_rope_args args, float *x) {
    const unsigned int head = blockIdx.x, row = blockIdx.y, lane = threadIdx.x & 31u;
    const float theta = (float)(args.start + row * args.stride) * args.frequencies[lane];
    const float c = cosf(theta);
    const float s = args.inverse ? -sinf(theta) : sinf(theta);
    const unsigned long long i = ((unsigned long long)row * args.heads + head) * args.width +
                                 args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16_dev(re * c - im * s);
    x[i + 1u] = dsv41_bf16_dev(re * s + im * c);
}

__global__ static void dsv41_quantize_kernel(unsigned int width, unsigned int rows, unsigned int mode, float *x) {
    const unsigned int block = mode == 3u ? 16u : 32u;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int column = blockIdx.x * block + lane;
    const bool valid = lane < block && column < width;
    const unsigned long long index = (unsigned long long)blockIdx.y * width + column;
    const float value = valid ? dsv41_bf16_dev(x[index]) : 0.0f;
    const float amax = dsv41_warp_allmax_f32(fabsf(value));
    float result = value;
    if (mode == 1u) {
        const float scale = dsv41_pow2_ceil_dev(fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysignf(dsv4_e4m3fn_dequant_dev(fabsf(value) / scale), value) * scale;
    } else if (mode == 2u || mode == 3u) {
        const float scale = mode == 3u
            ? dsv4_e4m3fn_dequant_dev(fmaxf(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil_dev(fmaxf(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysignf(dsv4_e2m1fn_dequant_dev(fabsf(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16_dev(result);
}

__global__ static void dsv41_engram_add_kernel(unsigned int width, unsigned int rows, float eps, unsigned int masked,
                                               float *residual, const float *kv, const float *q_weight,
                                               const float *k_weight, const unsigned char *mask) {
    const unsigned int gx = blockIdx.x, gy = blockIdx.y, lane = threadIdx.x & 31u;
    if (masked && !mask[gx]) return;
    const unsigned long long offset = ((unsigned long long)gx * 4u + gy) * width;
    const unsigned long long key_offset = ((unsigned long long)gx * 5u + gy) * width;
    const unsigned long long value_offset = ((unsigned long long)gx * 5u + 4u) * width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (unsigned int i = lane; i < width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16_dev(kv[key_offset + i]);
        const unsigned int wi = gy * width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = dsv41_warp_allsum_f32(h2);
    k2 = dsv41_warp_allsum_f32(k2);
    dot = dsv41_warp_allsum_f32(dot) * rsqrtf(h2 / width + eps) *
          rsqrtf(k2 / width + eps) * rsqrtf((float)width);
    const float gate = 1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)), dot)));
    for (unsigned int i = lane; i < width; i += 32u)
        residual[offset + i] = dsv41_bf16_dev(residual[offset + i] + gate * dsv41_bf16_dev(kv[value_offset + i]));
}

__global__ static void dsv41_pool2_kernel(unsigned int width, unsigned int pairs, unsigned int tail,
                                          float *out, const float *kv, const float *scores,
                                          const float *previous_kv, const float *previous_scores) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= width || y >= pairs) return;
    const long long a = (long long)y * 2 - (long long)tail;
    const unsigned long long b = (unsigned long long)(a + 1) * width + x;
    const float ka = a < 0 ? previous_kv[x] : kv[(unsigned long long)a * width + x];
    const float sa = a < 0 ? previous_scores[x] : scores[(unsigned long long)a * width + x];
    const float sb = scores[b], peak = fmaxf(sa, sb);
    const float ea = expf(sa - peak), eb = expf(sb - peak);
    out[(unsigned long long)y * width + x] = dsv41_bf16_dev((ka * ea + kv[b] * eb) / (ea + eb));
}

__global__ static void dsv41_candidate_blocks_kernel(unsigned int width, unsigned int rows, unsigned int start,
                                                     unsigned int ratio, const float *scores, float *blocks) {
    const unsigned int count = (width + 7u) / 8u;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= count || y >= rows) return;
    const unsigned int visible = min(width, (start + y + 1u) / ratio);
    float best = -INFINITY;
    for (unsigned int i = x * 8u; i < min(visible, (x + 1u) * 8u); i++)
        best = fmaxf(best, scores[(unsigned long long)y * width + i]);
    if (visible && x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(unsigned long long)y * count + x] = best;
}

__global__ static void dsv41_candidate_filter_kernel(unsigned int width, unsigned int rows, unsigned int start,
                                                     unsigned int ratio, const float *scores, float *out,
                                                     const float *block_mask) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= width || y >= rows) return;
    const unsigned long long offset = (unsigned long long)y * width + x;
    const unsigned int blocks = (width + 7u) / 8u;
    const unsigned int visible = min(width, (start + y + 1u) / ratio);
    out[offset] = (x < visible && block_mask[(unsigned long long)y * blocks + x / 8u] == 0.0f)
        ? scores[offset] : -INFINITY;
}

__global__ static void dsv41_carry_copy_kernel(unsigned int width, unsigned int rows, unsigned int words,
                                               unsigned int format, unsigned int pack,
                                               unsigned int *packed, float *plain) {
    const unsigned int tid = threadIdx.x, lane = tid & 31u;
    const unsigned int col = blockIdx.x * 128u + tid;
    const unsigned long long row = blockIdx.y;
    (void)rows;
    if (format == 0u) {
        if (col >= width) return;
        unsigned short *p = (unsigned short *)(packed + row * words);
        if (pack) p[col] = (unsigned short)(__float_as_uint(plain[row * width + col]) >> 16);
        else plain[row * width + col] = __uint_as_float(((unsigned int)p[col]) << 16);
    } else {
        const unsigned int word = col / 32u;
        if (pack) {
            const bool allowed = col < width && plain[row * width + col] == 0.0f;
            const unsigned int bits = __ballot_sync(0xffffffffu, allowed);
            if (!lane && word < words) packed[row * words + word] = bits;
        } else if (col < width) {
            const unsigned int bits = packed[row * words + word];
            plain[row * width + col] = (bits & (1u << lane)) ? 0.0f : -INFINITY;
        }
    }
}

__global__ static void dsv41_gather_kv_kernel(float *out, const float *source, const int *ids, unsigned int selected_rows) {
    const unsigned int r = blockIdx.x;
    if (r >= selected_rows) return;
    const float *src = source + (unsigned long long)(unsigned int)ids[r] * 512u;
    float *dst = out + (unsigned long long)r * 512u;
    for (unsigned int c = threadIdx.x; c < 512u; c += blockDim.x) dst[c] = src[c];
}

/* ---- host wrappers ---- */
static inline int dsv41_tensor_has_floats(const ds4_gpu_tensor *t, uint64_t n) {
    return t && t->bytes >= n * sizeof(float);
}

extern "C" int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                         uint32_t rows, uint32_t start, uint32_t stride,
                                         bool compressed, bool inverse) {
    if (width < 64 || !heads || !rows || rows > 1048576 || !stride ||
        (uint64_t)start + (uint64_t)(rows - 1u) * stride >= 1048576u ||
        (uint64_t)heads * rows > UINT64_MAX / width ||
        !dsv41_tensor_has_floats(x, (uint64_t)width * heads * rows)) return 0;
    static float frequencies[2][32];
    static int frequencies_ready = 0;
    if (!frequencies_ready) {   /* same table as ds4_metal.m (YaRN ramp for the compressed kind) */
        for (int kind = 0; kind < 2; kind++) {
            const float base = kind ? 160000.0f : 10000.0f;
            const float low = (float)floor(64.0 * log(65536.0 / (32.0 * 2.0 * M_PI)) / (2.0 * log(base)));
            const float high = (float)ceil(64.0 * log(65536.0 / (2.0 * M_PI)) / (2.0 * log(base)));
            for (int i = 0; i < 32; i++) {
                const float denominator = powf(base, (float)i / 32.0f);
                float f = 1.0f / denominator;
                if (kind) {
                    const float ramp = fminf(1.0f, fmaxf(0.0f, (i - low) / (high - low)));
                    const float smooth = 1.0f - ramp;
                    f = (f / 16.0f) * (1.0f - smooth) + f * smooth;
                }
                frequencies[kind][i] = f;
            }
        }
        frequencies_ready = 1;
    }
    dsv41_rope_args args;
    args.width = width; args.heads = heads; args.rows = rows; args.start = start;
    args.inverse = inverse ? 1u : 0u; args.stride = stride;
    memcpy(args.frequencies, frequencies[compressed ? 1 : 0], sizeof(args.frequencies));
    dsv41_rope_kernel<<<dim3(heads, rows, 1), 32, 0, cuda_decode_stream()>>>(args, (float *)x->ptr);
    return cuda_ok(cudaGetLastError(), "V4.1 unit-magnitude RoPE");
}

extern "C" int ds4_gpu_dsv41_rope(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                  uint32_t rows, uint32_t start, bool compressed, bool inverse) {
    return ds4_gpu_dsv41_rope_stride(x, width, heads, rows, start, 1, compressed, inverse);
}

extern "C" int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width, uint32_t rows,
                                      ds4_v41_activation_format format) {
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    if (!width || !rows || format < DS4_V41_BF16 || format > DS4_V41_FP4_E4M3 ||
        (format != DS4_V41_BF16 && width % block) ||
        !dsv41_tensor_has_floats(x, (uint64_t)width * rows)) return 0;
    const uint64_t count = (uint64_t)width * rows;
    if (format == DS4_V41_BF16) {
        const uint64_t quads = (count + 3u) / 4u;
        dsv41_bf16_linear_kernel<<<(unsigned)((quads + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (unsigned int *)x->ptr, (unsigned long long)count);
    } else {
        dsv41_quantize_kernel<<<dim3((unsigned)(((uint64_t)width + block - 1u) / block), rows, 1), 32, 0,
                                cuda_decode_stream()>>>(width, rows, (unsigned)format, (float *)x->ptr);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 activation quantization");
}

extern "C" int ds4_gpu_dsv41_engram_add(ds4_gpu_tensor *residual, const ds4_gpu_tensor *kv,
                                        const ds4_gpu_tensor *q_weight, const ds4_gpu_tensor *k_weight,
                                        const ds4_gpu_tensor *mask, uint32_t width, uint32_t rows, float eps) {
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || !isfinite(eps) || eps <= 0 || count > UINT64_MAX / 5u ||
        !dsv41_tensor_has_floats(residual, count * 4u) ||
        !dsv41_tensor_has_floats(kv, count * 5u) ||
        !dsv41_tensor_has_floats(q_weight, (uint64_t)width * 4u) ||
        !dsv41_tensor_has_floats(k_weight, (uint64_t)width * 4u) ||
        (mask && mask->bytes < rows)) return 0;
    dsv41_engram_add_kernel<<<dim3(rows, 4, 1), 32, 0, cuda_decode_stream()>>>(
        width, rows, eps, mask != NULL, (float *)residual->ptr, (const float *)kv->ptr,
        (const float *)q_weight->ptr, (const float *)k_weight->ptr,
        (const unsigned char *)(mask ? mask->ptr : residual->ptr));
    return cuda_ok(cudaGetLastError(), "V4.1 Engram gate");
}

extern "C" int ds4_gpu_dsv41_pool2(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *scores,
                                   ds4_gpu_tensor *previous_kv, ds4_gpu_tensor *previous_scores,
                                   uint32_t width, uint32_t rows, uint32_t start) {
    const uint64_t count = (uint64_t)width * rows;
    const uint32_t pairs = (uint32_t)(((uint64_t)rows + (start & 1u)) / 2u);
    if (!width || !rows || rows > UINT32_MAX - start ||
        !dsv41_tensor_has_floats(kv, count) || !dsv41_tensor_has_floats(scores, count) ||
        !dsv41_tensor_has_floats(previous_kv, width) ||
        !dsv41_tensor_has_floats(previous_scores, width) ||
        (pairs && !dsv41_tensor_has_floats(out, (uint64_t)width * pairs))) return 0;
    if (pairs) {
        const unsigned threads = width < 256u ? width : 256u;
        dsv41_pool2_kernel<<<dim3((width + threads - 1u) / threads, pairs, 1), threads, 0, cuda_decode_stream()>>>(
            width, pairs, start & 1u, (float *)out->ptr, (const float *)kv->ptr, (const float *)scores->ptr,
            (const float *)previous_kv->ptr, (const float *)previous_scores->ptr);
        if (!cuda_ok(cudaGetLastError(), "V4.1 KV pair pooling")) return 0;
    }
    /* Keep the last even-position input, as the row-at-a-time path does (Metal twin). */
    const uint32_t last_even = (start + rows - 1u) & ~1u;
    if (last_even >= start) {
        const uint64_t bytes = (uint64_t)width * sizeof(float);
        const uint64_t offset = (last_even - start) * bytes;
        if (!ds4_gpu_tensor_copy(previous_kv, 0, kv, offset, bytes) ||
            !ds4_gpu_tensor_copy(previous_scores, 0, scores, offset, bytes)) return 0;
    }
    return 1;
}

static int dsv41_candidates(ds4_gpu_tensor *out, const ds4_gpu_tensor *scores, const ds4_gpu_tensor *mask,
                            uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    if (!width || width > UINT32_MAX - 7u || !rows || !ratio || rows > UINT32_MAX - start) return 0;
    const uint32_t blocks = (width + 7u) / 8u;
    const uint32_t output_width = mask ? width : blocks;
    if (!dsv41_tensor_has_floats(scores, (uint64_t)width * rows) ||
        !dsv41_tensor_has_floats(out, (uint64_t)output_width * rows) ||
        (mask && !dsv41_tensor_has_floats(mask, (uint64_t)blocks * rows))) return 0;
    const unsigned threads = output_width < 256u ? output_width : 256u;
    const dim3 grid((output_width + threads - 1u) / threads, rows, 1);
    if (mask) {
        dsv41_candidate_filter_kernel<<<grid, threads, 0, cuda_decode_stream()>>>(
            width, rows, start, ratio, (const float *)scores->ptr, (float *)out->ptr, (const float *)mask->ptr);
    } else {
        dsv41_candidate_blocks_kernel<<<grid, threads, 0, cuda_decode_stream()>>>(
            width, rows, start, ratio, (const float *)scores->ptr, (float *)out->ptr);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 candidate selection");
}

extern "C" int ds4_gpu_dsv41_candidate_blocks(ds4_gpu_tensor *blocks, const ds4_gpu_tensor *scores,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return dsv41_candidates(blocks, scores, NULL, width, rows, start, ratio);
}

extern "C" int ds4_gpu_dsv41_candidate_filter(ds4_gpu_tensor *scores, const ds4_gpu_tensor *block_mask,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    if (!block_mask) return 0;
    return dsv41_candidates(scores, scores, block_mask, width, rows, start, ratio);
}

extern "C" int ds4_gpu_dsv41_carry_copy(ds4_gpu_tensor *packed, uint32_t row_offset, ds4_gpu_tensor *plain,
                                        uint32_t width, uint32_t rows, uint32_t format, bool pack) {
    if (!width || !rows || rows > UINT32_MAX - row_offset || format > DS4_V41_CARRY_MASK || packed == plain) return 0;
    const uint32_t words = format == DS4_V41_CARRY_BF16 ?
        (uint32_t)(((uint64_t)width + 1u) / 2u) : (uint32_t)(((uint64_t)width + 31u) / 32u);
    if (!dsv41_tensor_has_floats(packed, (uint64_t)(row_offset + rows) * words) ||
        !dsv41_tensor_has_floats(plain, (uint64_t)rows * width)) return 0;
    unsigned int *packed_ptr = (unsigned int *)packed->ptr + (uint64_t)row_offset * words;
    dsv41_carry_copy_kernel<<<dim3((unsigned)(((uint64_t)width + 127u) / 128u), rows, 1), 128, 0, cuda_decode_stream()>>>(
        width, rows, words, format, pack ? 1u : 0u, packed_ptr, (float *)plain->ptr);
    return cuda_ok(cudaGetLastError(), "V4.1 compact prefill carry");
}

extern "C" int ds4_gpu_dsv41_gather_kv(ds4_gpu_tensor *out, const ds4_gpu_tensor *source,
                                       const ds4_gpu_tensor *ids, uint32_t source_rows, uint32_t selected_rows) {
    if (!source_rows || !selected_rows || selected_rows > 512 || selected_rows > source_rows ||
        !dsv41_tensor_has_floats(source, (uint64_t)source_rows * 512u) ||
        !dsv41_tensor_has_floats(out, (uint64_t)selected_rows * 512u) ||
        !dsv41_tensor_has_floats(ids, selected_rows)) return 0;
    dsv41_gather_kv_kernel<<<selected_rows, 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)source->ptr, (const int *)ids->ptr, selected_rows);
    return cuda_ok(cudaGetLastError(), "V4.1 sparse KV gather");
}

extern "C" int ds4_gpu_dsv41_projection_rows(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint32_t width, uint32_t outputs,
                                             uint32_t rows, const ds4_gpu_tensor *in) {
    if (!width || !outputs || !rows || rows > 8192u || !model_map) return 0;
    return ds4_gpu_matmul_f16_tensor(out, model_map, model_size, weight_offset, width, outputs, in, rows);
}

/* Full-head prefill output: Q8 out_a -> BF16-rounded low -> Q8 out_b (the impl hook lives in ds4_cuda.cu). */
static int ds4_gpu_attention_output_q8_batch_impl_cuda(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, ds4_gpu_tensor *group_tmp, ds4_gpu_tensor *low_tmp,
        const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim,
        const ds4_gpu_tensor *heads, uint32_t n_tokens, int round_low_bf16);

extern "C" int ds4_gpu_dsv41_attention_output_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    return ds4_gpu_attention_output_q8_batch_impl_cuda(out, low, low, low, model_map, model_size,
        out_a_offset, out_b_offset, 4096, 1024, 8, 5120, heads, n_tokens, 1);
}

extern "C" int ds4_gpu_dsv41_attention_output_tp_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset, const ds4_gpu_tensor *heads, uint32_t n_tokens,
        uint32_t tp_rank) {
    (void)out; (void)low; (void)model_map; (void)model_size; (void)out_a_offset; (void)out_b_offset;
    (void)heads; (void)n_tokens; (void)tp_rank;
    return 0;   /* tensor parallelism is not part of the CUDA port */
}

/* Unpacked indexer path (the Metal tensor-op packed path is reported unavailable below). */
extern "C" int ds4_gpu_dsv41_indexer_scores_batch(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
                                                  const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
                                                  uint32_t source_rows, uint32_t rows, uint32_t start, uint32_t ratio) {
    if (ratio != 1u && ratio != 2u) return 0;
    return ds4_gpu_indexer_scores_decode_batch_tensor(scores, q, weights, keys, source_rows, rows, start,
                                                      32u, 128u, ratio, 1.0f / 64.0f);
}

extern "C" int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
                                                uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !rows || rows > UINT32_MAX - start ||
        width > INT32_MAX || rows > INT32_MAX || (start + rows) / ratio > width ||
        (start + 1u) / ratio < 1024u) return 0;
    return ds4_gpu_indexer_topk_tensor(selected, scores, width, rows, 512u);
}

extern "C" int ds4_gpu_dsv41_tensor_ops_available(void) { return 0; }

extern "C" uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t source_rows, uint32_t rows) {
    /* Only the flag words; the packed bf16 operands are never used on CUDA (tensor ops off). */
    return (((uint64_t)rows + ((uint64_t)source_rows + 63u) / 64u) * 4u + 255u) & ~UINT64_C(255);
}

extern "C" int ds4_gpu_dsv41_indexer_pack(ds4_gpu_tensor *packed, const ds4_gpu_tensor *q, const ds4_gpu_tensor *keys,
                                          uint32_t source_rows, uint32_t rows) {
    (void)packed; (void)q; (void)keys; (void)source_rows; (void)rows;
    return 0;
}

extern "C" int ds4_gpu_dsv41_indexer_scores_packed(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
                                                   const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
                                                   const ds4_gpu_tensor *packed, uint32_t source_rows, uint32_t rows,
                                                   uint32_t start, uint32_t ratio, uint32_t packed_rows, uint32_t offset) {
    (void)scores; (void)q; (void)weights; (void)keys; (void)packed; (void)source_rows; (void)rows;
    (void)start; (void)ratio; (void)packed_rows; (void)offset;
    return 0;
}

/* mode-0 definition of the Metal fused wrapper: rms-norm the rows, then the f16 projection. */
extern "C" int ds4_gpu_hc_rms_scale_project_f16_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *scale_scratch, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x,
        uint32_t n_rows, float eps) {
    if (!out || !scale_scratch || !model_map || !x || !in_dim || !out_dim || !n_rows || n_rows > INT32_MAX) return 0;
    return ds4_gpu_rms_norm_plain_rows_tensor(scale_scratch, x, in_dim, n_rows, eps) != 0 &&
           ds4_gpu_matmul_f16_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim,
                                     scale_scratch, n_rows) != 0;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(
        const ds4_gpu_stream_expert_table *table, const int32_t *expert_ids,
        const uint32_t *expert_priorities, uint32_t n_experts) {
    /* Mirrors ds4_gpu_stream_expert_cache_seed_experts on CUDA: the CUDA expert cache seeds itself. */
    (void)table; (void)expert_ids; (void)expert_priorities; (void)n_experts;
    return 1;
}

extern "C" int ds4_gpu_tp_failed(void) { return 0; }

/* ---- hooks for ds4_cuda.cu ---- */
static int dsv41_round_low_bf16_launch(ds4_gpu_tensor *low, uint64_t count) {
    if (!low || low->bytes < count * sizeof(float)) return 0;
    const uint64_t quads = (count + 3u) / 4u;
    dsv41_bf16_linear_kernel<<<(unsigned)((quads + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (unsigned int *)low->ptr, (unsigned long long)count);
    return cuda_ok(cudaGetLastError(), "V4.1 low BF16 rounding");
}

extern "C" int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, ds4_gpu_tensor *group_tmp, ds4_gpu_tensor *low_tmp,
        const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    return ds4_gpu_attention_output_q8_batch_impl_cuda(out, low, group_tmp, low_tmp, model_map, model_size,
        out_a_offset, out_b_offset, group_dim, rank, n_groups, out_dim, heads, n_tokens, 0);
}

/* ---- general router (arbitrary expert count) ----
 * The V4 CUDA router is hard-locked to 256 experts / top-6 / scale 1.5. V4.1 routes 6 of 384,
 * so the shipped shape is kept on its tuned kernels and any other shape lands here. The math is
 * the reference router verbatim: p = sqrt(softplus(logit)), rank by p+bias, then renormalize the
 * selected weights and apply the expert scale. */
__global__ static void dsv41_router_select_general_kernel(
        int32_t *selected, float *weights, float *probs,
        const float *bias, const float *logits,
        uint32_t n_expert, uint32_t n_expert_used, float scale, int has_bias) {
    /* SPARKPORT: the reference selection is an insertion sort run by one
     * thread, 384 x 6 compares per token per layer = 197 us on the GB10, 8 ms
     * of every decode token.  This is the same selection as n_expert_used
     * rounds of block-wide argmax: largest score wins, equal scores go to
     * the LOWER expert index, exactly the order the insertion sort produced
     * (a later equal score never displaces an earlier one).  The weights are
     * then summed in the same j order, so the result is bit-identical. */
    const uint32_t t = blockIdx.x;
    const float *log_row = logits + (uint64_t)t * n_expert;
    float *prob = probs + (uint64_t)t * n_expert;
    int32_t *sel = selected + (uint64_t)t * n_expert_used;
    float *w = weights + (uint64_t)t * n_expert_used;
    extern __shared__ float sprob[];          /* [n_expert] p, then [n_expert] live score */
    float *sscore = sprob + n_expert;
    __shared__ float s_wval[32];
    __shared__ uint32_t s_widx[32];
    __shared__ int32_t s_sel[64];
    for (uint32_t i = threadIdx.x; i < n_expert; i += blockDim.x) {
        const float p = sqrtf(softplus_dev(log_row[i]));
        sprob[i] = p;
        prob[i] = p;
        sscore[i] = p + (has_bias ? bias[i] : 0.0f);
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t n_warps = (blockDim.x + 31u) >> 5u;
    const uint32_t rounds = n_expert_used < 64u ? n_expert_used : 64u;
    for (uint32_t j = 0; j < rounds; j++) {
        float best = -INFINITY;
        uint32_t best_i = 0xffffffffu;
        for (uint32_t i = threadIdx.x; i < n_expert; i += blockDim.x) {
            const float v = sscore[i];
            if (v > best || (v == best && i < best_i)) { best = v; best_i = i; }
        }
        for (uint32_t off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, best, off);
            const uint32_t oi = __shfl_down_sync(0xffffffffu, best_i, off);
            if (ov > best || (ov == best && oi < best_i)) { best = ov; best_i = oi; }
        }
        if (lane == 0) { s_wval[warp] = best; s_widx[warp] = best_i; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bv = -INFINITY;
            uint32_t bi = 0xffffffffu;
            for (uint32_t k = 0; k < n_warps; k++) {
                const float v = s_wval[k];
                const uint32_t i = s_widx[k];
                if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
            }
            s_sel[j] = bi == 0xffffffffu ? -1 : (int32_t)bi;
            if (bi != 0xffffffffu) sscore[bi] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    for (uint32_t j = 0; j < n_expert_used; j++) sel[j] = j < rounds ? s_sel[j] : -1;
    float sum = 0.0f;
    for (uint32_t j = 0; j < n_expert_used; j++) {
        const int32_t e = sel[j];
        const float v = (e >= 0 && (uint32_t)e < n_expert) ? sprob[e] : 0.0f;
        w[j] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (uint32_t j = 0; j < n_expert_used; j++) w[j] = w[j] / sum * scale;
}

static int dsv41_router_select_general(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
                                       ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size,
                                       uint64_t bias_offset, bool has_bias, const ds4_gpu_tensor *logits,
                                       uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale,
                                       uint32_t n_tokens) {
    if (!selected || !weights || !probs || !logits || !model_map || !n_tokens ||
        !n_expert || !n_expert_used || n_expert_used > n_expert || n_expert_used > 64u || n_expert > 8192u ||
        logits->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(float)) return 0;
    const float *bias = NULL;
    if (has_bias) {
        const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
        if (bias_offset > model_size || bias_bytes > model_size - bias_offset) return 0;
        bias = (const float *)cuda_resolve_weight_ptr(model_map, bias_offset, bias_bytes,
                                                      ds4_tensor_device_idx(selected), "router_bias");
        if (!bias) return 0;
    }
    const unsigned threads = n_expert < 1024u ? ((n_expert + 31u) & ~31u) : 1024u;
    dsv41_router_select_general_kernel<<<n_tokens, threads, 2u * n_expert * sizeof(float), cuda_decode_stream()>>>(
        (int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr, bias,
        (const float *)logits->ptr, n_expert, n_expert_used, expert_weight_scale, has_bias ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "V4.1 general router select");
}

extern "C" int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs,
        const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset,
        uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used,
        float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used,
        bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits) {
    if (n_expert == 256u && n_expert_used == 6u && fabsf(expert_weight_scale - 1.5f) <= 1.0e-6f)
        return ds4_gpu_router_select_tensor_v4_impl(selected, weights, probs, model_map, model_size,
            bias_offset, hash_offset, hash_rows, token, n_expert, n_expert_used, expert_weight_scale,
            n_expert_groups, n_group_used, has_bias, hash_mode, logits);
    if (hash_mode || n_expert_groups > 1u || n_group_used > 0u) return 0;
    return dsv41_router_select_general(selected, weights, probs, model_map, model_size, bias_offset,
                                       has_bias, logits, n_expert, n_expert_used, expert_weight_scale, 1u);
}

extern "C" int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs,
        const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset,
        uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias,
        bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens,
        uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens) {
    if (n_expert == 256u && n_expert_used == 6u && fabsf(expert_weight_scale - 1.5f) <= 1.0e-6f)
        return ds4_gpu_router_select_batch_tensor_v4_impl(selected, weights, probs, model_map, model_size,
            bias_offset, hash_offset, hash_rows, n_expert_groups, n_group_used, has_bias, hash_mode,
            logits, tokens, n_expert, n_expert_used, expert_weight_scale, n_tokens);
    if (hash_mode || n_expert_groups > 1u || n_group_used > 0u) return 0;
    return dsv41_router_select_general(selected, weights, probs, model_map, model_size, bias_offset,
                                       has_bias, logits, n_expert, n_expert_used, expert_weight_scale, n_tokens);
}
