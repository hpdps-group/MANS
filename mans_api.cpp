#include "mans_api.hpp"

#include <stdexcept>

#ifdef MANS_ENABLE_CPU
#include "cpu/mans_cpu.h"
#endif

#ifdef MANS_ENABLE_NV
#include "nv/mans_nv.h"
#endif

namespace mans {

void compress_device(const void* input_data,
                     size_t length,
                     const MansParams& params,
                     uint8_t* out,
                     size_t& out_size) {
    if (params.backend != Backend::NVIDIA) {
        throw std::runtime_error("MANS::compress_device: only the NVIDIA backend is supported.");
    }
#ifdef MANS_ENABLE_NV
    mans::nv::compress_internal_device(input_data, length, params, out, out_size);
    return;
#else
    throw std::runtime_error("MANS::compress_device: NVIDIA backend was NOT compiled.");
#endif
}

void decompress_device(const void* input_data,
                       size_t length,
                       const MansParams& params,
                       uint8_t* out,
                       size_t& out_size) {
    if (params.backend != Backend::NVIDIA) {
        throw std::runtime_error("MANS::decompress_device: only the NVIDIA backend is supported.");
    }
#ifdef MANS_ENABLE_NV
    mans::nv::decompress_internal_device(input_data, length, params, out, out_size);
    return;
#else
    throw std::runtime_error("MANS::decompress_device: NVIDIA backend was NOT compiled.");
#endif
}

void compress(const void* input_data,
              size_t length,
              const MansParams& params,
              uint8_t* out,
              size_t& out_size) {
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        mans::cpu::compress_internal(input_data, length, params, out, out_size, false, "");
        return;
#else
        throw std::runtime_error("MANS::compress: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        mans::nv::compress_internal(input_data, length, params, out, out_size, false, "");
        return;
#else
        throw std::runtime_error("MANS::compress: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::compress: Unknown backend type.");
}

void decompress(const void* input_data,
                size_t length,
                const MansParams& params,
                uint8_t* out,
                size_t& out_size) {
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        mans::cpu::decompress_internal(input_data, length, params, out, out_size, false, "");
        return;
#else
        throw std::runtime_error("MANS::decompress: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        mans::nv::decompress_internal(input_data, length, params, out, out_size, false, "");
        return;
#else
        throw std::runtime_error("MANS::decompress: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::decompress: Unknown backend type.");
}

std::size_t get_mans_max_compress_bytes(std::size_t num_elements, const MansParams& params) {
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        return mans::cpu::get_max_compress_bytes(num_elements, params);
#else
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        return mans::nv::get_max_compress_bytes(num_elements, params);
#else
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::get_mans_max_compress_bytes: Unknown backend type.");
}

std::size_t get_mans_exact_decompress_bytes(const void* compressed_data,
                                            std::size_t compressed_len,
                                            const MansParams& params) {
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        return mans::cpu::get_exact_decompress_bytes(compressed_data, compressed_len, params);
#else
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        return mans::nv::get_exact_decompress_bytes(compressed_data, compressed_len, params);
#else
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: Unknown backend type.");
}

} // namespace mans
