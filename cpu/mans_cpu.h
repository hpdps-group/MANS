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
    bool open_benchmark
);

void compress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path,
    bool open_benchmark,
    adm::AdmCompressScratch* adm_scratch,
    bool reuse_scratch
);


void decompress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path,
    bool open_benchmark
);

void decompress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path,
    bool open_benchmark,
    adm::AdmDecompressScratch* adm_scratch,
    bool reuse_scratch
);

}
}