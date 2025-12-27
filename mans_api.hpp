#pragma once
#include <vector>
#include <string>
#include <stdexcept>


#include "mans_defs.h"
#include "cpu/mans_cpu.h"
// #include "nv/mans_nv.cuh"

namespace mans {

// top module: Compress
inline void compress(
    const void* input_data, 
    size_t length, 
    const MansParams& params, 
    std::vector<uint8_t>& out
) {
    if (params.backend == Backend::CPU) {
        // set all debug signal to false at release 
        mans::cpu::compress_internal(input_data, length, params, out, false, "", false);
        return;
    }
    if (params.backend == Backend::NVIDIA) {
        throw std::runtime_error("mans::compress: NVIDIA backend is not implemented");
    }
    throw std::runtime_error("mans::compress: unknown/unsupported backend");
}

// top module: Decompress
inline void decompress(
    const void* input_data, 
    size_t length,
    const MansParams& params, 
    std::vector<uint8_t>& out
) {
    if (params.backend == Backend::CPU) {
        // set all debug signal to false at release 
        mans::cpu::decompress_internal(input_data, length, params, out, false, "", false);
        return;
    }
    if (params.backend == Backend::NVIDIA) {
        throw std::runtime_error("mans::decompress: NVIDIA backend is not implemented");
    }
    throw std::runtime_error("mans::decompress: unknown/unsupported backend");
}

} // namespace mans