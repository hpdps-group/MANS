#pragma once
#include <vector>
#include <string>
#include "../mans_defs.h" 
namespace adm {
struct AdmCompressScratch;
struct AdmDecompressScratch;
}
namespace mans {
namespace cpu {


void compress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path,
    adm::AdmCompressScratch* adm_scratch = nullptr,
    bool reuse_scratch = false,
    std::uint8_t* mans_intermediate_buf = nullptr,
    std::size_t mans_intermediate_cap = 0
);


void decompress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path,
    adm::AdmDecompressScratch* adm_scratch = nullptr,
    bool reuse_scratch = false,
    std::uint8_t* mans_intermediate_buf = nullptr,
    std::size_t mans_intermediate_cap = 0
);

}
}