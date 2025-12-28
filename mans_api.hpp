#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include "mans_defs.h"

namespace mans {


void compress(
    const void* input_data, 
    size_t length, 
    const MansParams& params, 
    std::vector<uint8_t>& out
);


void decompress(
    const void* input_data, 
    size_t length,
    const MansParams& params, 
    std::vector<uint8_t>& out
);

} // namespace mans