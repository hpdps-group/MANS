#include "mans_cpu.h"
#include <iostream>
#include <cstring>
#include <limits>
#include <algorithm>
#include <vector>


#include "adm/adm_utils.h"
#include "pans/pans_utils.h"
#include "file_utils.h"

namespace mans {
namespace cpu {

// ==========================================
// 1.  Compress Helper Function
// ==========================================

template<typename T>
static bool decide_use_adm(const T* data, size_t size, uint32_t threshold) {
    const std::size_t block_size = 512;
    std::uint64_t max_block_diff = 0;

    for (std::size_t i = 0; i < size; i += block_size) {
        std::size_t end = std::min(i + block_size, size);
        T bmin = std::numeric_limits<T>::max();
        T bmax = std::numeric_limits<T>::min();

        for (std::size_t j = i; j < end; ++j) {
            T v = data[j];
            if (v < bmin) bmin = v;
            if (v > bmax) bmax = v;
        }

        std::uint64_t diff = static_cast<std::uint64_t>(bmax) - static_cast<std::uint64_t>(bmin);
        if (diff > max_block_diff) {
            max_block_diff = diff;
        }
        if (max_block_diff > threshold) return false;
    }
    return (max_block_diff <= threshold);
}

static void prepend_header(
    const std::vector<std::uint8_t>& payload,
    std::vector<std::uint8_t>& final_payload,
    std::uint8_t codec)
{
    final_payload.clear();
    final_payload.reserve(1 + payload.size());
    final_payload.push_back(codec);
    final_payload.insert(final_payload.end(), payload.begin(), payload.end());
}

// ==========================================
// 2. Decompress Helper Function
// ==========================================

static bool parse_header(
    const uint8_t* data, 
    size_t length, 
    uint8_t& codec)
{
    if (length < sizeof(MansHeader)) {
        std::cerr << "[Error] File too small, invalid mans format.\n";
        return false;
    }
    // read first byte as codec
    codec = data[0]; 
    return true;
}

// ==========================================
// 3. Core Compress/Decompress Loginic
// ==========================================

template<typename T>
void do_compress_t(const T* data_ptr, size_t length, const MansParams& params, 
                   std::vector<uint8_t>& final_out, 
                   bool save_adm, const std::string& dump_path, bool open_benchmark) {
    
    uint32_t threshold = params.adm_threshold; 
    if (threshold == 0) threshold = 4000; 

    bool use_adm = decide_use_adm(data_ptr, length, threshold);

    std::vector<uint8_t> pans_input;
    uint8_t codec_code = 0;

    if (use_adm) {
        codec_code = 1; // ADM
        std::vector<T> temp_vec(data_ptr, data_ptr + length);
        if (open_benchmark){
            adm_compress_and_benchmark(temp_vec, pans_input);
        } else {
            adm_compress(temp_vec, pans_input);
        }
        
        if (save_adm && !dump_path.empty()) {
            save_u8_file(dump_path, pans_input);
        }
    } else {
        codec_code = 2; // Direct
        pans_input.resize(length * sizeof(T));
        if (length > 0) {
            std::memcpy(pans_input.data(), data_ptr, length * sizeof(T));
        }
    }

    std::vector<uint8_t> pans_output;
    if (open_benchmark) {
        pans_compress_and_benchmark(
            pans_input,
            pans_output
        );
    } else {
        uint32_t bs=0, cs=0; // dummy vars if required by signature
        double dur = 0.0;
        pans_compress(
            pans_input,
            pans_output,
            bs,
            cs,
            dur 
        );
    }
    

    prepend_header(pans_output, final_out, codec_code);
}

template<typename T>
void do_decompress_t(const uint8_t* input_ptr, size_t length,
                     std::vector<uint8_t>& final_out,
                     bool save_adm, const std::string& dump_path, bool open_benchmark)
{
    
    uint8_t codec = 0;
    if (!parse_header(input_ptr, length, codec)) {
        return;
    }

    // skip the 1-byte header
    const uint8_t* payload_ptr = input_ptr + 1;
    size_t payload_len = length - 1;

    
    std::vector<uint8_t> payload_vec(payload_ptr, payload_ptr + payload_len);

    // 2. PANS Decompress
    // The result of PANS may be the final data (Codec 2), or it may be ADM-compressed data (Codec 1)
    std::vector<uint8_t> pans_data;

    if (open_benchmark) {
        pans_decompress_and_benchmark(payload_vec, pans_data);
    } else {
        uint32_t bs=0, cs=0; // dummy vars if required by signature
        double dur = 0.0;
        pans_decompress(payload_vec, pans_data, bs, cs,dur); 
    }

    if (codec == 2) {
        // === Direct Mode ===
        // PANS decompression output is the original data (byte stream)
        final_out = std::move(pans_data);
    } 
    else if (codec == 1) {
        // === ADM Mode ===
        
        // Debug: Save ADM compressed data
        if (save_adm && !dump_path.empty()) {
            save_u8_file(dump_path, pans_data);
        }

        // ADM Decompress 
        std::vector<T> recovered_items;
        if (open_benchmark) {
            adm_decompress_and_benchmark(pans_data, recovered_items);
        } else {
            adm_decompress(pans_data, recovered_items);
        }

        // Convert: vector<T> -> vector<uint8_t> (API requires output as a byte stream)
        size_t total_bytes = recovered_items.size() * sizeof(T);
        final_out.resize(total_bytes);
        if (total_bytes > 0) {
            std::memcpy(final_out.data(), recovered_items.data(), total_bytes);
        }
    } 
    else {
        std::cerr << "[Error] Unknown codec type: " << int(codec) << "\n";
    }
}

// ==========================================
// 5. Exposed implementation interface
// ==========================================

void compress_internal(const void* input_data, size_t length, const MansParams& params, 
                       std::vector<uint8_t>& out, 
                       bool save_adm, const std::string& dump_path, bool open_benchmark) {
    if (params.dtype == DataType::U16) {
        do_compress_t(static_cast<const uint16_t*>(input_data), length, params, out, save_adm, dump_path, open_benchmark);
    } else if (params.dtype == DataType::U32) {
        do_compress_t(static_cast<const uint32_t*>(input_data), length, params, out, save_adm, dump_path, open_benchmark);
    }
}


void decompress_internal(const void* input_data, size_t length, const MansParams& params, 
                         std::vector<uint8_t>& out, 
                         bool save_adm, const std::string& dump_path, bool open_benchmark) {


    const uint8_t* ptr = static_cast<const uint8_t*>(input_data);

    if (params.dtype == DataType::U16) {
        do_decompress_t<uint16_t>(ptr, length, out, save_adm, dump_path, open_benchmark);
    } else if (params.dtype == DataType::U32) {
        do_decompress_t<uint32_t>(ptr, length, out, save_adm, dump_path, open_benchmark);
    }
}

} // namespace cpu
} // namespace mans