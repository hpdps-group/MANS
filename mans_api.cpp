#include "mans_api.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <limits>
#include <vector>


#ifdef MANS_ENABLE_CPU
    #include "cpu/mans_cpu.h"
    #include "cpu/adm/adm_utils.h"
    #include "cpu/adm/adm.h"
    #include "cpu/pans/CpuANSUtils.h"
    #include "cpu/pans/pans_utils.h"
    #include "cpu/fse/fse_utils.h"
    extern "C" {
        #include "cpu/fse/include/fse.h"
    }
#endif

#ifdef MANS_ENABLE_NV
    // #include "nv/mans_nv.h"
#endif

namespace mans {

void compress(const void* input_data,
              size_t length,
              const MansParams& params,
              uint8_t* out,
              size_t& out_size) {
    // 1. CPU branch
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        mans::cpu::compress_internal(input_data,
                                     length,
                                     params,
                                     out,
                                     out_size,
                                     false,
                                     "");
        return;
#else
        throw std::runtime_error("MANS::compress: CPU backend was NOT compiled.");
#endif
    }

    // 2. NV branch
    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        // mans::nv::compress_internal(...);
        throw std::runtime_error("MANS::compress: NVIDIA backend not implemented yet.");
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
        mans::cpu::decompress_internal(input_data,
                                       length,
                                       params,
                                       out,
                                       out_size,
                                       false,
                                       "");
        return;
#else
        throw std::runtime_error("MANS::decompress: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        // mans::nv::decompress_internal(...);
        throw std::runtime_error("MANS::decompress: NVIDIA backend not implemented yet.");
        return;
#else
        throw std::runtime_error("MANS::decompress: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::decompress: Unknown backend type.");
}

std::size_t get_mans_max_compress_bytes(std::size_t num_elements, const MansParams& params) {
    if (num_elements == 0) {
        return 0;
    }

    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        std::size_t elem_size = 0;
        if (params.dtype == DataType::U16) {
            elem_size = sizeof(std::uint16_t);
        } else if (params.dtype == DataType::U32) {
            elem_size = sizeof(std::uint32_t);
        } else {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes: Unsupported dtype.");
        }

        const std::size_t max_u32 = std::numeric_limits<uint32_t>::max();
        if (num_elements > max_u32 / elem_size) {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes: raw_bytes exceeds 32-bit limit.");
        }
        const std::size_t raw_bytes = num_elements * elem_size;
        if (raw_bytes > max_u32) {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes: raw_bytes exceeds 32-bit limit.");
        }

        std::size_t adm_bytes = 0;
        if (params.dtype == DataType::U16) {
            adm_bytes = adm_max_compressed_size<std::uint16_t>(num_elements);
        } else {
            adm_bytes = adm_max_compressed_size<std::uint32_t>(num_elements);
        }
        if (adm_bytes > max_u32) {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes: adm_bytes exceeds 32-bit limit.");
        }

        const std::size_t pans_raw =
            static_cast<std::size_t>(cpu_ans::getMaxCompressedSize(static_cast<uint32_t>(raw_bytes)));
        const std::size_t pans_adm =
            static_cast<std::size_t>(cpu_ans::getMaxCompressedSize(static_cast<uint32_t>(adm_bytes)));
        const std::size_t fse_raw = FSE_compressBound(raw_bytes);
        const std::size_t fse_adm = FSE_compressBound(adm_bytes);
        if (fse_raw == 0 || fse_adm == 0) {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes: FSE_compressBound overflow.");
        }

        const std::uint32_t mode = (params.mode == Mode::R) ? Mode::R : Mode::P;
        if (mode == Mode::P) {
            return 1 + std::max(pans_raw, pans_adm); // 1 byte for codec header
        }
        if (mode == Mode::R) {
            return 1 + std::max(fse_raw, fse_adm); // 1 byte for codec header
        }
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: Unknown mode.");
#else
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: NVIDIA backend not implemented yet.");
#else
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::get_mans_max_compress_bytes: Unknown backend type.");
}

std::size_t get_mans_exact_decompress_bytes(const void* compressed_data,
                                            std::size_t compressed_len,
                                            const MansParams& params) {
    if (!compressed_data) {
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: compressed_data is null.");
    }
    if (compressed_len < 1) {
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: compressed_len too small.");
    }

    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        const auto* bytes = static_cast<const std::uint8_t*>(compressed_data);
        const std::uint8_t codec = bytes[0];
        if (compressed_len <= 1) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: missing payload.");
        }

        const std::uint8_t* payload_ptr = bytes + 1;
        const std::size_t payload_len = compressed_len - 1;
        const std::uint32_t mode = (params.mode == Mode::R) ? Mode::R : Mode::P;

        std::size_t stage2_compress_len = 0;
        std::size_t stage2_decompress_len = 0;
        if (mode == Mode::P) {
            stage2_compress_len = payload_len;
            get_compress_and_decompressed_len(payload_ptr, stage2_compress_len, stage2_decompress_len);
            if (stage2_decompress_len == 0 || stage2_compress_len != payload_len) {
                throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: failed to parse PANS payload.");
            }
        } else if (mode == Mode::R) {
            std::string parse_error;
            if (!mans::cpu::get_fse_compress_and_decompressed_len(payload_ptr, payload_len,
                                                                  stage2_compress_len,
                                                                  stage2_decompress_len,
                                                                  &parse_error)) {
                throw std::runtime_error(
                    "MANS::get_mans_exact_decompress_bytes: failed to parse FSE payload: " +
                    parse_error);
            }
            if (stage2_compress_len != payload_len) {
                throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: FSE payload truncated.");
            }
        } else {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: unknown mode.");
        }

        if (codec == 2) {
            return stage2_decompress_len; // direct payload -> raw bytes
        }
        if (codec != 1) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: unknown codec.");
        }

        std::vector<std::uint8_t> adm_blob(stage2_decompress_len);
        std::size_t adm_blob_len = stage2_decompress_len;
        double stage2_dur = 0.0;
        if (mode == Mode::P) {
            pans_decompress(payload_ptr, payload_len, adm_blob.data(), adm_blob_len, stage2_dur);
        } else {
            mans::cpu::fse_decompress(payload_ptr, payload_len, adm_blob.data(), adm_blob_len, stage2_dur);
        }
        if (adm_blob_len < sizeof(adm::FileHeader)) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: ADM header too small.");
        }

        const auto* hdr = reinterpret_cast<const adm::FileHeader*>(adm_blob.data());
        std::size_t elem_size = 0;
        if (params.dtype == DataType::U16) {
            elem_size = sizeof(std::uint16_t);
        } else if (params.dtype == DataType::U32) {
            elem_size = sizeof(std::uint32_t);
        } else {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: Unsupported dtype.");
        }

        const std::size_t max_size = std::numeric_limits<std::size_t>::max();
        if (hdr->num_elements > max_size / elem_size) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: size overflow.");
        }
        return static_cast<std::size_t>(hdr->num_elements) * elem_size;
#else
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: NVIDIA backend not implemented yet.");
#else
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: Unknown backend type.");
}

} // namespace mans
