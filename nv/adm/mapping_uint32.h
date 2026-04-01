#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace mans {
namespace nv {
namespace adm {

void compress_u32_device(const std::uint32_t* d_input,
                         std::size_t num_elements,
                         std::uint8_t* d_output,
                         std::size_t& output_size,
                         cudaStream_t stream = nullptr);

void decompress_u32_device(const std::uint8_t* d_input,
                           std::size_t input_size,
                           std::uint32_t* d_output,
                           std::size_t num_elements,
                           cudaStream_t stream = nullptr);

std::size_t get_max_u32_payload_bytes(std::size_t num_elements);

} // namespace adm
} // namespace nv
} // namespace mans
