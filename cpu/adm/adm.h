// algorithm.h
#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <memory>

#include <immintrin.h>
#include <omp.h>
#include "../../mans_defs.h"
#include "../buffer_cache.h"

#ifdef ENABLE_TIMING
#include "../../mans_timing.h"
#endif
#ifndef MANS_TIMING_SCOPE
#define MANS_TIMING_SCOPE(name) do {} while (0)
#endif

namespace adm {

// ---------------- global parameters ----------------
inline constexpr int cmp_tblock_size = 32;
inline constexpr int cmp_chunk = 16;
inline constexpr int cmp_block_x = 16;
inline constexpr int cmp_block_y = 16;
inline constexpr int cmp_block_z = 16;
inline constexpr int decmp_chunk = 32;
inline constexpr int max_bytes_signal_per_ele_16b = 2;
inline constexpr int max_bytes_signal_per_ele_32b = 3;
inline constexpr int warp_size = 32;

// ------------- header -------------
// record metadata
struct FileHeader {
    std::uint64_t num_elements; // uint16 elements num
    std::uint64_t gsize;        // warp = ceil(num / (cmp_tblock_size * cmp_chunk))
    std::size_t len1;
    std::size_t len2;
    std::size_t len3;
    std::size_t len4;    

};

inline void compress_uint16(
    const uint16_t* input_data,
    size_t input_len,
    // int dims,                 // 1/2/3
    // int nx,
    // int ny,                   // if dims<2, ny=0
    // int nz,                   // if dims<3, nz=0
    int* output_lengths,
    uint16_t* centers,
    uint8_t* codes,
    uint8_t* bit_signals,
    size_t& bit_signals_len,
    const mans::MansParams& params
) {
    int dims = params.dims;
    int nx = params.nx;
    int ny = params.ny;
    int nz = params.nz;
    if (!input_data || !output_lengths || !centers || !codes || !bit_signals) return;
    if (dims < 1 || dims > 3) return;
    if (nx <= 0) return;
    if (dims >= 2 && ny <= 0) return;
    if (dims == 3 && nz <= 0) return;

    constexpr int warp_threads = cmp_tblock_size; // usually 32
    constexpr int chunk_1d = cmp_chunk;           // 1D: each lane handles 16 (original)

    // 2D/3D tile size (you said 16 if dimension exists)
    constexpr int blk_x = 16;
    constexpr int blk_y = 16;
    constexpr int blk_z = 16;

    const int ny_eff = (dims >= 2) ? ny : 1;
    const int nz_eff = (dims == 3) ? nz : 1;

    const size_t num_elements = (size_t)nx * (size_t)ny_eff * (size_t)nz_eff;
    const size_t safe_elements = std::min(input_len, num_elements);

    auto idx3 = [&](int x, int y, int z) -> size_t {
        return (size_t)x + (size_t)y * (size_t)nx + (size_t)z * (size_t)nx * (size_t)ny_eff;
    };

    // ---- block grid / gsize ----
    int gsize = 0;
    int grid_x = 0, grid_y = 1, grid_z = 1;
    int block_elems_max = 0;

    if (dims == 1) {
        // keep original: 32*16
        block_elems_max = warp_threads * chunk_1d; // 512
        gsize = (int)((safe_elements + block_elems_max - 1) / block_elems_max);
        grid_x = gsize; grid_y = 1; grid_z = 1; // logical
    } else if (dims == 2) {
        grid_x = (nx + blk_x - 1) / blk_x;
        grid_y = (ny_eff + blk_y - 1) / blk_y;
        grid_z = 1;
        gsize = grid_x * grid_y;
        block_elems_max = blk_x * blk_y; // 256
    } else { // dims == 3
        grid_x = (nx + blk_x - 1) / blk_x;
        grid_y = (ny_eff + blk_y - 1) / blk_y;
        grid_z = (nz_eff + blk_z - 1) / blk_z;
        gsize = grid_x * grid_y * grid_z;
        block_elems_max = blk_x * blk_y * blk_z; // 4096
    }
    const int total_threads = gsize * warp_threads;

    int* signal_length = nullptr;
    int* bit_offsets = nullptr;
    uint8_t* tmp_bit_signals = nullptr;

    const int elems_per_thread_max = (block_elems_max + warp_threads - 1) / warp_threads;
    const int bytes_per_thread = elems_per_thread_max * max_bytes_signal_per_ele_16b;
    const size_t tmp_bytes = (size_t)total_threads * (size_t)bytes_per_thread;    
    auto& cache = mans::cpu::BufferCache::instance();
    signal_length = cache.get_t<int>("adm_u16_signal_length", static_cast<std::size_t>(gsize));
    bit_offsets = cache.get_t<int>("adm_u16_bit_offsets", static_cast<std::size_t>(total_threads));
    tmp_bit_signals = cache.get_t<uint8_t>("adm_u16_tmp_bit_signals", tmp_bytes);
    if (!signal_length || !bit_offsets || !tmp_bit_signals) {
        std::cerr << "Failed to allocate ADM scratch buffers.\n";
        return;
    }

    auto block_to_coords = [&](int b, int& bx, int& by, int& bz) {
        if (dims == 1) { bx = b; by = 0; bz = 0; return; }
        if (dims == 2) {
            bx = b % grid_x;
            by = b / grid_x;
            bz = 0;
            return;
        }
        // dims == 3
        bx = b % grid_x;
        int t = b / grid_x;
        by = t % grid_y;
        bz = t / grid_y;
    };

    // =========================================================
    // center_calc: 1D uses contiguous segment; 2D/3D uses tile neighborhood
    // =========================================================
    {
        MANS_TIMING_SCOPE("adm/compress/center_calc");
        #pragma omp parallel for num_threads(params.adm_center_calc_threads)
        for (int b = 0; b < gsize; ++b) {
            uint64_t sum = 0;
            uint64_t cnt = 0;

            if (dims == 1) {
                const int base = b * (warp_threads * chunk_1d);
                const int end  = std::min(base + warp_threads * chunk_1d, (int)safe_elements);
                for (int i = base; i < end; ++i) { sum += input_data[i]; }
                cnt = (uint64_t)(end - base);
            } else {
                int bx_i, by_i, bz_i;
                block_to_coords(b, bx_i, by_i, bz_i);

                const int x0 = bx_i * blk_x;
                const int x1 = std::min(x0 + blk_x, nx);
                const int y0 = by_i * blk_y;
                const int y1 = std::min(y0 + blk_y, ny_eff);
                const int z0 = bz_i * blk_z;
                const int z1 = std::min(z0 + blk_z, nz_eff);

                for (int z = z0; z < z1; ++z) {
                    for (int y = y0; y < y1; ++y) {
                        const size_t base = idx3(x0, y, z);
                        if (base >= safe_elements) continue;
                        const int len = x1 - x0;
                        const int safe_len = (int)std::min((size_t)len, safe_elements - base);
                        const uint16_t* p = input_data + base;
                        for (int i = 0; i < safe_len; ++i) sum += p[i];
                        cnt += (uint64_t)safe_len;
                    }
                }
            }

            centers[b] = (cnt > 0) ? (uint16_t)(sum / cnt) : 0;
        }
    }

    {
        MANS_TIMING_SCOPE("adm/compress/encode");
    // Encoding and setting codes, bit_signals (in temporary space)
    #pragma omp parallel for num_threads(params.adm_encode_threads)
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int lane = thread_idx % cmp_tblock_size;
        int base_idx = warp * block_elems_max + lane * elems_per_thread_max;

        uint8_t* bit_out = &tmp_bit_signals[thread_idx * bytes_per_thread];
        std::memset(bit_out, 0, bytes_per_thread);

        if (base_idx >= num_elements) {
            bit_offsets[thread_idx] = 0;
            continue;
        }
        int center = centers[warp];

        int bit_offset = 0;

        for (int i = 0; i < elems_per_thread_max && base_idx + i < num_elements; ++i) {
            uint16_t val = input_data[base_idx + i];
            int diff = val > center ? val - center : center - val;
            int output_len = (val == center) ? 1 : (diff + 125) / 126;
            uint8_t res = (val == center) ? 1 : ((diff + 126 - output_len * 126) * 2 + (val > center ? -1 : 0) + 1);

            codes[base_idx + i] = res;

            // Set bitstream (mark the corresponding bit)
            bit_out[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            bit_offset += output_len;
        }

        bit_offsets[thread_idx] = bit_offset;
    }
    }

    {
        MANS_TIMING_SCOPE("adm/compress/encode");
        #pragma omp parallel for num_threads(params.adm_encode_threads)
        for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
            const int b    = thread_idx / warp_threads;
            const int lane = thread_idx % warp_threads;

            uint8_t* bit_out = &tmp_bit_signals[thread_idx * bytes_per_thread];
            std::memset(bit_out, 0, bytes_per_thread);

            int center = centers[b];
            int bit_offset = 0;

            if (dims == 1) {
                // keep original mapping: lane handles contiguous 16 values
                const int base = b * (warp_threads * chunk_1d) + lane * chunk_1d;
                if (base >= (int)safe_elements) { bit_offsets[thread_idx] = 0; continue; }

                const int end = std::min(base + chunk_1d, (int)safe_elements);
                for (int idx = base; idx < end; ++idx) {
                    uint16_t val = input_data[idx];
                    int diff = val > center ? val - center : center - val;
                    int output_len = (val == center) ? 1 : (diff + 125) / 126;
                    uint8_t res = (val == center) ? 1 : ((diff + 126 - output_len * 126) * 2 + (val > center ? -1 : 0) + 1);

                    codes[idx] = res;

                    // Set bitstream (mark the corresponding bit)
                    bit_out[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
                    bit_offset += output_len;
                }

                bit_offsets[thread_idx] = bit_offset;
                continue;
            }

            // dims=2/3: tile -> flatten in-tile order (x fastest, then y, then z)
            int bx_i, by_i, bz_i;
            block_to_coords(b, bx_i, by_i, bz_i);

            const int x0 = bx_i * blk_x;
            const int x1 = std::min(x0 + blk_x, nx);
            const int y0 = by_i * blk_y;
            const int y1 = std::min(y0 + blk_y, ny_eff);
            const int z0 = bz_i * blk_z;
            const int z1 = std::min(z0 + blk_z, nz_eff);

            const int sx = x1 - x0;
            const int sy = y1 - y0;
            const int sz = z1 - z0;
            const int block_elems = sx * sy * sz;
            if (block_elems <= 0) { bit_offsets[thread_idx] = 0; continue; }

            // contiguous slice assignment (compatible style with 1D)
            const int per_lane = (block_elems + warp_threads - 1) / warp_threads; // ceil
            const int k0 = lane * per_lane;
            const int k1 = std::min(k0 + per_lane, block_elems);
            if (k0 >= k1) { bit_offsets[thread_idx] = 0; continue; }

            const int plane = sx * sy;
            for (int k = k0; k < k1; ++k) {
                const int lz = (dims == 3) ? (k / plane) : 0;
                const int rem = (dims == 3) ? (k - lz * plane) : k;
                const int ly = rem / sx;
                const int lx = rem - ly * sx;

                const int x = x0 + lx;
                const int y = y0 + ly;
                const int z = z0 + lz;

                const size_t idx = idx3(x, y, z);
                if (idx >= safe_elements) continue;

                uint16_t val = input_data[idx];
                int diff = val > center ? val - center : center - val;
                int output_len = (val == center) ? 1 : (diff + 125) / 126;
                uint8_t res = (val == center) ? 1 : ((diff + 126 - output_len * 126) * 2 + (val > center ? -1 : 0) + 1);

                codes[idx] = res;

                // Set bitstream (mark the corresponding bit)
                bit_out[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
                bit_offset += output_len;
            }

            bit_offsets[thread_idx] = bit_offset;
        }
    }

    {
        MANS_TIMING_SCOPE("adm/compress/warp_reduce");
    // Warp-level reduction: compute signal_length[warp] deterministically
    #pragma omp parallel for num_threads(params.adm_warp_reduce_threads)
    for (int warp = 0; warp < gsize; ++warp) {
        int base_thread = warp * warp_threads;
        int end_thread = std::min(base_thread + warp_threads, total_threads);

        int max_len_bytes = 0;
        for (int t = base_thread; t < end_thread; ++t) {
            int bit_offset = bit_offsets[t];
            int length_bytes = (bit_offset + 7) / 8;
            max_len_bytes = std::max(max_len_bytes, length_bytes);
        }

        signal_length[warp] = max_len_bytes;
    }
    }

    {
        MANS_TIMING_SCOPE("adm/compress/fill_tail");
    // Fill in the tail bits
    #pragma omp parallel for num_threads(params.adm_fill_tail_threads)
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / warp_threads;
        int bit_offset = bit_offsets[thread_idx];
        int max_len_bytes = signal_length[warp];
        if (bit_offset >= max_len_bytes * 8) continue;

        uint8_t* bit_out = &tmp_bit_signals[thread_idx * bytes_per_thread];
        int byte_idx = bit_offset / 8;
        uint8_t mask = (bit_offset % 8 == 0) ? 0xFF : (0xFF >> (bit_offset % 8));
        bit_out[byte_idx] |= mask;

        // for (int bb = byte_idx + 1; bb < max_len_bytes; ++bb) {
        //     if (bb < bytes_per_thread) bit_out[bb] = 0xFF;
        // }  
    }
    }

    {
        MANS_TIMING_SCOPE("adm/compress/prefix_sum");
    // Compute prefix sum (serially)
    output_lengths[0] = 0;
    for (int i = 1; i <= gsize; ++i) {
        output_lengths[i] = output_lengths[i - 1] + signal_length[i - 1];
    }
    }

    // Write back bit_signals
    int total_bit_bytes = output_lengths[gsize] * warp_threads;
    bit_signals_len = static_cast<size_t>(total_bit_bytes);

    {
        MANS_TIMING_SCOPE("adm/compress/write_back");
    #pragma omp parallel for num_threads(params.adm_write_back_threads)
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / warp_threads;
        int lane = thread_idx % warp_threads;
        int bit_len = signal_length[warp];
        int dst_base = output_lengths[warp] * warp_threads + lane * bit_len;

        if (dst_base + bit_len > total_bit_bytes) continue;

        const uint8_t* src = &tmp_bit_signals[thread_idx * bytes_per_thread];
        // 使用向量化指令进行批量拷贝
        #pragma omp simd
        for (int i = 0; i < bit_len; ++i) {
            bit_signals[dst_base + i] = src[i];
        }
    }
    }
}

inline void decompress_uint16(
    const int* output_lengths,           
    size_t gsize,                        
    const uint16_t* centers,             
    const uint8_t* codes,                
    size_t num_elements,                
    const uint8_t* bit_signals,          
    uint16_t* output_data,
    const mans::MansParams& params
)
{
    int total_threads = gsize * cmp_tblock_size;

    // Step 1: Restore signal[]
    uint8_t* signals = nullptr;

    {
        MANS_TIMING_SCOPE("adm_alloc_scratch");
        signals = mans::cpu::BufferCache::instance().get_t<uint8_t>(
            "adm_u16_signals", num_elements);
    }
    if (!signals) {
        std::cerr << "Failed to allocate ADM scratch buffer.\n";
        return;
    }

    {
        MANS_TIMING_SCOPE("adm/decompress/restore_signals");
    #pragma omp parallel for num_threads(params.adm_restore_signals_threads)
    for (int tid = 0; tid < total_threads; ++tid) {
        int warp = tid / cmp_tblock_size;
        int lane = tid % cmp_tblock_size;
        int idx = tid;

        if (idx * cmp_chunk >= num_elements) continue;

        int length = output_lengths[warp + 1] - output_lengths[warp];

        int src_start_idx = output_lengths[warp] * cmp_tblock_size + lane * length;
        int dst_start_idx = idx * cmp_chunk;

        uint8_t bit_buffer = 0;
        int signal_idx = -1;
        int offset_byte = 0;
        bool bit = 0;

        uint8_t local_signal[cmp_chunk] = {0};

        for (; offset_byte < length && signal_idx < cmp_chunk; offset_byte++) {
            bit_buffer = bit_signals[src_start_idx + offset_byte];
            for (int i = 7; i >= 0 && signal_idx < cmp_chunk; i--) {
                bit = (bit_buffer >> i) & 1;
                if (bit) {
                    signal_idx++;
                } else {
                    local_signal[signal_idx]++;
                }
            }
        }

        // Use a local copy to avoid accessing shared memory repeatedly
        for (int i = 0; i < cmp_chunk && dst_start_idx + i < num_elements; ++i) {
            signals[dst_start_idx + i] = local_signal[i];
        }
    }
    }

    {
        MANS_TIMING_SCOPE("adm/decompress/decode_values");
    #pragma omp parallel for num_threads(params.adm_decode_values_threads)
    for (int tid = 0; tid < total_threads; ++tid) {
        int block_id = tid;
        int lane = block_id % warp_size;
        int bid = block_id / cmp_tblock_size;
        int base_idx = block_id * decmp_chunk;

        if (base_idx >= num_elements) continue;

        uint16_t center = (lane < 16) ? centers[bid * 2] : centers[bid * 2 + 1];

        // Use local variables to minimize memory access and reduce branch conditions
        for (int i = 0; i < decmp_chunk && base_idx + i < num_elements; ++i) {
            uint8_t code = codes[base_idx + i];
            uint8_t signal = signals[base_idx + i];

            int diff = (code % 2 == 1) ? ((code - 1) / 2) : (code / 2);
            diff += signal * 126;

            uint16_t val = (code % 2 == 1) ? center - diff : center + diff;
            output_data[base_idx + i] = val;
        }
    }
    }
}

inline void compress_uint32(
    const uint32_t* input_data,          
    size_t input_len,                    
    int* output_lengths,
    uint32_t* centers,
    uint8_t* codes,
    uint8_t* bit_signals,
    size_t& bit_signals_len,
    const mans::MansParams& params
) {
    int num_elements = (int)input_len;   
    int gsize = (num_elements + cmp_tblock_size * cmp_chunk - 1) / (cmp_tblock_size * cmp_chunk);
    int total_threads = gsize * cmp_tblock_size;

    int* signal_length = nullptr;
    int* bit_offsets = nullptr;
    uint8_t* tmp_bit_signals = nullptr;

    const int bytes_per_thread = cmp_chunk * max_bytes_signal_per_ele_32b;
    const size_t tmp_bytes = static_cast<size_t>(total_threads) * bytes_per_thread;

    {
        MANS_TIMING_SCOPE("adm_alloc_scratch");
        auto& cache = mans::cpu::BufferCache::instance();
        signal_length = cache.get_t<int>("adm_u32_signal_length", static_cast<std::size_t>(gsize));
        bit_offsets = cache.get_t<int>("adm_u32_bit_offsets", static_cast<std::size_t>(total_threads));
        tmp_bit_signals = cache.get_t<uint8_t>("adm_u32_tmp_bit_signals", tmp_bytes);
    }
    if (!signal_length || !bit_offsets || !tmp_bit_signals) {
        std::cerr << "Failed to allocate ADM scratch buffers.\n";
        return;
    }

    // static const uint8_t bitmask[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
    // static const uint8_t tail_mask[8] = {0xFF, 0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01};

    // Center calculation: parallelizing and reducing unnecessary work
    #pragma omp parallel for num_threads(params.adm_center_calc_threads)
    for (int warp = 0; warp < gsize; ++warp) {
        int base_idx = warp * cmp_tblock_size * cmp_chunk;
        int end_idx = std::min(base_idx + cmp_tblock_size * cmp_chunk, num_elements);

        uint64_t sum = 0;
        for (int i = base_idx; i < end_idx; ++i) {
            sum += input_data[i];
        }

        int count = end_idx - base_idx;
        centers[warp] = (count > 0) ? sum / count : 0;
    }

    // Allocate temporary buffer for bit_signals
    // Encoding and setting codes, bit_signals (in temporary space)
    #pragma omp parallel for num_threads(params.adm_encode_threads)
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int lane = thread_idx % cmp_tblock_size;
        int base_idx = warp * cmp_tblock_size * cmp_chunk + lane * cmp_chunk;

        uint8_t* bit_out = &tmp_bit_signals[thread_idx * bytes_per_thread];
        std::memset(bit_out, 0, bytes_per_thread);

        if (base_idx >= num_elements) {
            bit_offsets[thread_idx] = 0;
            continue;
        }
        int center = centers[warp];

        int bit_offset = 0;

        for (int i = 0; i < cmp_chunk && base_idx + i < num_elements; ++i) {
            uint32_t val = input_data[base_idx + i];
            int diff = val > center ? val - center : center - val;
            int output_len = (val == center) ? 1 : (diff + 125) / 126;
            uint8_t res = (val == center) ? 1 : ((diff + 126 - output_len * 126) * 2 + (val > center ? -1 : 0) + 1);

            codes[base_idx + i] = res;

            // Set bitstream (mark the corresponding bit)
            // bit_out[bit_offset / 8] |= bitmask[bit_offset % 8];
            bit_out[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            bit_offset += output_len;
        }

        bit_offsets[thread_idx] = bit_offset;
    }

    // Warp-level reduction: compute signal_length[warp] deterministically
    #pragma omp parallel for num_threads(params.adm_warp_reduce_threads)
    for (int warp = 0; warp < gsize; ++warp) {
        int base_thread = warp * cmp_tblock_size;
        int end_thread = std::min(base_thread + cmp_tblock_size, total_threads);

        int max_len_bytes = 0;
        for (int t = base_thread; t < end_thread; ++t) {
            int bit_offset = bit_offsets[t];
            int length_bytes = (bit_offset + 7) / 8;
            max_len_bytes = std::max(max_len_bytes, length_bytes);
        }

        signal_length[warp] = max_len_bytes;
    }

    // Fill in the tail bits
    #pragma omp parallel for num_threads(params.adm_fill_tail_threads)
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int bit_offset = bit_offsets[thread_idx];
        int max_len_bytes = signal_length[warp];

        if (bit_offset < max_len_bytes * 8) {
            uint8_t* bit_out = &tmp_bit_signals[thread_idx * cmp_chunk * max_bytes_signal_per_ele_32b];
            int byte_idx = bit_offset / 8;
            uint8_t mask = (bit_offset % 8 == 0) ? 0xFF : (0xFF >> (bit_offset % 8));
            bit_out[byte_idx] |= mask;
            // bit_out[byte_idx] |= tail_mask[bit_offset % 8];
        }
    }

    // Compute prefix sum (serially)
    output_lengths[0] = 0;
    for (int i = 1; i <= gsize; ++i) {
        output_lengths[i] = output_lengths[i - 1] + signal_length[i - 1];
    }

    // Write back bit_signals
    int total_bit_bytes = output_lengths[gsize] * cmp_tblock_size;
    bit_signals_len = static_cast<size_t>(total_bit_bytes);

    #pragma omp parallel for num_threads(params.adm_write_back_threads)
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int lane = thread_idx % cmp_tblock_size;
        int bit_len = signal_length[warp];
        int dst_base = output_lengths[warp] * cmp_tblock_size + lane * bit_len;

        if (dst_base + bit_len > total_bit_bytes) continue;

        const uint8_t* src = &tmp_bit_signals[thread_idx * cmp_chunk * max_bytes_signal_per_ele_32b];
        // Use  simd for bulk copying
        #pragma omp simd
        for (int i = 0; i < bit_len; ++i) {
            bit_signals[dst_base + i] = src[i];
        }
    }
}

inline void decompress_uint32(
    const int* output_lengths,           
    size_t gsize,                        
    const uint32_t* centers,             
    const uint8_t* codes,                
    size_t num_elements,                 
    const uint8_t* bit_signals,          
    uint32_t* output_data,
    const mans::MansParams& params
)
{
    int total_threads = gsize * cmp_tblock_size;

    // Step 1: Restore signal[]
    uint8_t* signals = nullptr;

    MANS_TIMING_SCOPE("adm_alloc_scratch");
    signals = mans::cpu::BufferCache::instance().get_t<uint8_t>(
        "adm_u32_signals", num_elements);
    if (!signals) {
        std::cerr << "Failed to allocate ADM scratch buffer.\n";
        return;
    }

    {
        MANS_TIMING_SCOPE("adm/decompress/restore_signals");
    #pragma omp parallel for num_threads(params.adm_restore_signals_threads)
    for (int tid = 0; tid < total_threads; ++tid) {
        int warp = tid / cmp_tblock_size;
        int lane = tid % cmp_tblock_size;
        int idx = tid;

        if (idx * cmp_chunk >= num_elements) continue;

        int length = output_lengths[warp + 1] - output_lengths[warp];

        int src_start_idx = output_lengths[warp] * cmp_tblock_size + lane * length;
        int dst_start_idx = idx * cmp_chunk;

        uint8_t bit_buffer = 0;
        int signal_idx = -1;
        int offset_byte = 0;
        bool bit = 0;

        uint8_t local_signal[cmp_chunk] = {0};

        for (; offset_byte < length && signal_idx < cmp_chunk; offset_byte++) {
            bit_buffer = bit_signals[src_start_idx + offset_byte];
            for (int i = 7; i >= 0 && signal_idx < cmp_chunk; i--) {
                bit = (bit_buffer >> i) & 1;
                if (bit) {
                    signal_idx++;
                } else {
                    local_signal[signal_idx]++;
                }
            }
        }

        // Use a local copy to avoid accessing shared memory repeatedly
        for (int i = 0; i < cmp_chunk && dst_start_idx + i < num_elements; ++i) {
            signals[dst_start_idx + i] = local_signal[i];
        }
    }
    }

    // Step 2: Decode values
    {
        MANS_TIMING_SCOPE("adm/decompress/decode_values");
    #pragma omp parallel for num_threads(params.adm_decode_values_threads)
    for (int tid = 0; tid < total_threads; ++tid) {
        int block_id = tid;
        int lane = block_id % warp_size;
        int bid = block_id / cmp_tblock_size;
        int base_idx = block_id * decmp_chunk;

        if (base_idx >= num_elements) continue;

        uint32_t center = (lane < 16) ? centers[bid * 2] : centers[bid * 2 + 1];

        // Use local variables to minimize memory access and reduce branch conditions
        for (int i = 0; i < decmp_chunk && base_idx + i < num_elements; ++i) {
            uint8_t code = codes[base_idx + i];
            uint8_t signal = signals[base_idx + i];

            int diff = (code % 2 == 1) ? ((code - 1) / 2) : (code / 2);
            diff += signal * 126;

            uint32_t val = (code % 2 == 1) ? center - diff : center + diff;
            output_data[base_idx + i] = val;
        }
        }
    }
}

} // namespace adm

#endif // ALGORITHM_H
