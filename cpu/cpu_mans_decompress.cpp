#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "mans_cpu.h"
#include "file_utils.h"
#include "../mans_defs.h"

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <-u2|-u4> <input_file> <output_file>\n";
}

std::uint64_t read_le64(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(p[0]) |
           (static_cast<std::uint64_t>(p[1]) << 8) |
           (static_cast<std::uint64_t>(p[2]) << 16) |
           (static_cast<std::uint64_t>(p[3]) << 24) |
           (static_cast<std::uint64_t>(p[4]) << 32) |
           (static_cast<std::uint64_t>(p[5]) << 40) |
           (static_cast<std::uint64_t>(p[6]) << 48) |
           (static_cast<std::uint64_t>(p[7]) << 56);
}

bool parse_raw_bytes(const std::vector<std::uint8_t>& input, std::size_t& raw_bytes) {
    if (input.size() < mans::kMansHeaderBytes) {
        std::cerr << "Input is too small to be a MANS file.\n";
        return false;
    }
    mans::MansHeader header{};
    std::memcpy(&header, input.data(), sizeof(header));
    const std::uint64_t raw = read_le64(header.raw_bytes_le);
    if (raw == 0 || raw > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        std::cerr << "Invalid raw byte size in MANS header.\n";
        return false;
    }
    raw_bytes = static_cast<std::size_t>(raw);
    return true;
}

template <typename T>
bool save_output_file(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    if constexpr (std::is_same<T, std::uint16_t>::value) {
        std::vector<std::uint16_t> typed(bytes.size() / sizeof(std::uint16_t));
        std::memcpy(typed.data(), bytes.data(), bytes.size());
        return save_u16_file(path, typed);
    } else {
        std::vector<std::uint32_t> typed(bytes.size() / sizeof(std::uint32_t));
        std::memcpy(typed.data(), bytes.data(), bytes.size());
        return save_u32_file(path, typed);
    }
}

template <typename T>
bool run_decompress(const std::string& input_file,
                    const std::string& output_file,
                    mans::MansParams params) {
    std::vector<std::uint8_t> input;
    if (!load_u8_file(input_file, input)) {
        std::cerr << "Failed to load input file: " << input_file << "\n";
        return false;
    }

    std::size_t raw_bytes = 0;
    if (!parse_raw_bytes(input, raw_bytes)) {
        return false;
    }
    if ((raw_bytes % sizeof(T)) != 0) {
        std::cerr << "Raw byte size is incompatible with the requested dtype.\n";
        return false;
    }

    std::vector<std::uint8_t> output(raw_bytes);
    std::size_t output_size = output.size();
    mans::cpu::decompress_internal(input.data(), input.size(), params,
                                   output.data(), output_size,
                                   false, std::string());
    if (output_size == 0) {
        std::cerr << "Decompression failed.\n";
        return false;
    }
    if (output_size != raw_bytes) {
        std::cerr << "Unexpected decompressed size: " << output_size
                  << " expected " << raw_bytes << "\n";
        return false;
    }
    if (!save_output_file<T>(output_file, output)) {
        std::cerr << "Failed to write output file: " << output_file << "\n";
        return false;
    }

    std::cout << "Decompressed " << input_file << " -> " << output_file
              << " (" << output_size << " bytes)\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string dtype_flag = argv[1];
    const bool is_u2 = (dtype_flag == "-u2" || dtype_flag == "u2");
    const bool is_u4 = (dtype_flag == "-u4" || dtype_flag == "u4");
    if (!is_u2 && !is_u4) {
        print_usage(argv[0]);
        return 1;
    }

    mans::MansParams params{};
    params.backend = mans::Backend::CPU;
    params.dtype = is_u2 ? mans::DataType::U16 : mans::DataType::U32;

    const std::string input_file = argv[2];
    const std::string output_file = argv[3];

    if (is_u2) {
        return run_decompress<std::uint16_t>(input_file, output_file, params) ? 0 : 1;
    }
    return run_decompress<std::uint32_t>(input_file, output_file, params) ? 0 : 1;
}
