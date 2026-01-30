// compiler: g++ -std=c++17 -O3 cpu_mans_decompress.cpp mans_cpu.cpp -o cpu_mans_decompress -fopenmp
// exec    : OMP_NUM_THREADS=4 ./cpu_mans_decompress u2 input.bin output.u2 0
//           OMP_NUM_THREADS=4 ./cpu_mans_decompress u4 input.bin output.u4 1

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../mans_defs.h"
#include "../mans_timing.h"
#include "mans_cpu.h"
#include "mans_container.h"
#include "file_utils.h"

namespace {

constexpr const char* kAnsiReset = "\033[0m";
constexpr const char* kAnsiBold = "\033[1m";
constexpr const char* kAnsiDim = "\033[2m";
constexpr const char* kAnsiRed = "\033[31m";
constexpr const char* kAnsiGreen = "\033[32m";
constexpr const char* kAnsiYellow = "\033[33m";
constexpr const char* kAnsiBlue = "\033[34m";
constexpr const char* kAnsiCyan = "\033[36m";

struct ChunkMeta {
    std::size_t payload_offset = 0;
    std::size_t comp_len = 0;
    std::size_t raw_len = 0;
};

enum class ContainerParseResult {
    kNotContainer,
    kOk,
    kError
};

double read_input_with_timing(const std::string& input_file,
                              std::vector<std::uint8_t>& input_data) {
    MANS_TIMING_SCOPE("io_read");
    auto start = std::chrono::high_resolution_clock::now();
    bool ok = load_u8_file(input_file, input_data);
    auto end = std::chrono::high_resolution_clock::now();
    if (!ok) {
        return -1.0;
    }
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double write_output_with_timing(const std::string& output_file,
                                const std::uint8_t* data,
                                std::size_t size) {
    MANS_TIMING_SCOPE("io_write");
    auto start = std::chrono::high_resolution_clock::now();
    bool ok = save_u8_file(output_file, data, size);
    auto end = std::chrono::high_resolution_clock::now();
    if (!ok) {
        return -1.0;
    }
    return std::chrono::duration<double, std::milli>(end - start).count();
}

ContainerParseResult parse_container(
    const std::vector<std::uint8_t>& input,
    std::uint32_t expected_dtype,
    std::vector<ChunkMeta>& chunks,
    std::size_t& total_raw_bytes,
    std::uint64_t& chunk_count,
    std::string& error) {
    total_raw_bytes = 0;
    chunk_count = 0;
    chunks.clear();

    if (input.size() < sizeof(mans::container::ContainerHeader)) {
        return ContainerParseResult::kNotContainer;
    }

    mans::container::ContainerHeader header{};
    std::memcpy(&header, input.data(), sizeof(header));
    if (!mans::container::magic_matches(
            header.magic, mans::container::kContainerMagic,
            sizeof(mans::container::kContainerMagic))) {
        return ContainerParseResult::kNotContainer;
    }

    if (header.version != mans::container::kContainerVersion) {
        error = "Unsupported container version: " + std::to_string(header.version);
        return ContainerParseResult::kError;
    }
    if (header.header_bytes != sizeof(mans::container::ContainerHeader)) {
        error = "Container header size mismatch.";
        return ContainerParseResult::kError;
    }
    if (header.chunk_header_bytes != sizeof(mans::container::ChunkHeader)) {
        error = "Chunk header size mismatch.";
        return ContainerParseResult::kError;
    }

    const std::uint8_t expected_dtype_id =
        (expected_dtype == mans::DataType::U16) ? 1 : 2;
    if (header.dtype != expected_dtype_id) {
        error = "Data type mismatch: file dtype=" + std::to_string(header.dtype) +
                ", expected=" + std::to_string(expected_dtype_id);
        return ContainerParseResult::kError;
    }

    const std::uint64_t index_bytes =
        header.chunk_count * sizeof(mans::container::IndexEntry);
    if (header.index_offset + index_bytes > input.size()) {
        error = "Index table exceeds file size.";
        return ContainerParseResult::kError;
    }
    if (header.data_offset > input.size()) {
        error = "Data offset exceeds file size.";
        return ContainerParseResult::kError;
    }

    chunks.reserve(static_cast<std::size_t>(header.chunk_count));
    for (std::uint64_t i = 0; i < header.chunk_count; ++i) {
        const std::size_t index_offset = static_cast<std::size_t>(
            header.index_offset + i * sizeof(mans::container::IndexEntry));
        mans::container::IndexEntry entry{};
        std::memcpy(&entry, input.data() + index_offset, sizeof(entry));

        if (entry.offset > input.size() - sizeof(mans::container::ChunkHeader)) {
            error = "Chunk header offset exceeds file size.";
            return ContainerParseResult::kError;
        }
        const std::size_t chunk_offset = static_cast<std::size_t>(entry.offset);
        mans::container::ChunkHeader chunk_header{};
        std::memcpy(&chunk_header, input.data() + chunk_offset, sizeof(chunk_header));
        if (!mans::container::magic_matches(
                chunk_header.magic, mans::container::kChunkMagic,
                sizeof(mans::container::kChunkMagic))) {
            error = "Chunk header magic mismatch.";
            return ContainerParseResult::kError;
        }
        if (chunk_header.version != mans::container::kChunkVersion) {
            error = "Unsupported chunk version.";
            return ContainerParseResult::kError;
        }
        if (chunk_header.header_bytes != sizeof(mans::container::ChunkHeader)) {
            error = "Chunk header size mismatch.";
            return ContainerParseResult::kError;
        }
        if (chunk_header.comp_len != entry.comp_len ||
            chunk_header.raw_len != entry.raw_len) {
            error = "Chunk index mismatch.";
            return ContainerParseResult::kError;
        }

        const std::size_t payload_offset =
            chunk_offset + sizeof(mans::container::ChunkHeader);
        if (entry.comp_len > input.size() - payload_offset) {
            error = "Chunk payload exceeds file size.";
            return ContainerParseResult::kError;
        }

        chunks.push_back(
            ChunkMeta{payload_offset,
                      static_cast<std::size_t>(entry.comp_len),
                      static_cast<std::size_t>(entry.raw_len)});
        total_raw_bytes += static_cast<std::size_t>(entry.raw_len);
    }

    chunk_count = header.chunk_count;
    return ContainerParseResult::kOk;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << kAnsiRed << "Use: " << kAnsiReset << argv[0]
                  << " <u2|u4> <input_bin_file> <output_u2/u4_file> <save_adm>"
                  << " [--csv out.csv]\n";
        return 1;
    }

    std::string dtype_str   = argv[1];
    std::string input_file  = argv[2];
    std::string output_file = argv[3];
    std::string save_flag   = argv[4];
    bool save_adm = (save_flag == "1");
    std::string csv_path;

    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        }
    }
    // 1. build MansParams
    mans::MansParams params{};
    params.backend = mans::Backend::CPU;
    
    // Note: You must specify the target type (u2/u4) when decompressing,
    // otherwise the ADM cannot restore the correct values.
    if (dtype_str == "u2" || dtype_str == "-u2") {
        params.dtype = mans::DataType::U16;
    } else if (dtype_str == "u4" || dtype_str == "-u4") {
        params.dtype = mans::DataType::U32;
    } else {
        std::cerr << "Unknown data type flag: " << dtype_str << "\nUse: u2 or u4\n";
        return 1;
    }

    std::cout << kAnsiBold << "Command-line arguments:" << kAnsiReset << "\n";
    std::cout << "  " << kAnsiCyan << "Input type" << kAnsiReset << ": " << dtype_str << "\n";
    std::cout << "  " << kAnsiCyan << "Input file" << kAnsiReset << ": " << input_file << "\n";
    std::cout << "  " << kAnsiCyan << "Output file" << kAnsiReset << ": " << output_file << "\n";
    std::cout << "  " << kAnsiCyan << "Save ADM" << kAnsiReset << ": "
              << (save_adm ? kAnsiGreen : kAnsiYellow)
              << (save_adm ? "yes" : "no") << kAnsiReset << "\n";
    if (!csv_path.empty()) {
        std::cout << "  " << kAnsiCyan << "CSV output" << kAnsiReset << ": " << csv_path << "\n";
    }
    std::cout << "\n";

    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        if (!csv) {
            std::cerr << "Failed to open CSV output: " << csv_path << "\n";
            return 1;
        }
        csv << "input_bytes,output_bytes,chunk_count,decomp_ms,io_read_ms,io_write_ms,"
               "throughput_mbps,io_ratio\n";
    }

    constexpr int kIters = 11;
    double total_decomp_ms = 0.0;
    double total_io_read_ms = 0.0;
    double total_io_write_ms = 0.0;
    std::size_t input_bytes = 0;
    std::size_t output_bytes_size = 0;
    std::uint64_t chunk_count = 0;
    bool used_container = false;

    for (int iter = 0; iter < kIters; ++iter) {
        MANS_TIMING_RUN_SCOPE();
        MANS_TIMING_SCOPE("total");
        std::vector<uint8_t> input_data;
        std::unique_ptr<std::uint8_t, decltype(&free)> output_bytes(nullptr, &free);
        double io_read_ms = 0.0;
        double io_write_ms = 0.0;
        double decomp_ms = 0.0;
        std::size_t output_write_size = 0;

        io_read_ms = read_input_with_timing(input_file, input_data);
        if (io_read_ms < 0.0) {
            std::cerr << kAnsiRed << "Failed to load input file: " << kAnsiReset
                      << input_file << "\n";
            return 1;
        }
        if (input_data.empty()) {
            std::cerr << kAnsiRed << "Input file is empty." << kAnsiReset << "\n";
            return 1;
        }
        input_bytes = input_data.size();

        std::vector<ChunkMeta> chunks;
        std::size_t total_raw_bytes = 0;
        std::string error;
        ContainerParseResult parse_result =
            parse_container(input_data, params.dtype, chunks, total_raw_bytes, chunk_count, error);
        if (parse_result == ContainerParseResult::kError) {
            std::cerr << kAnsiRed << "Invalid container format: " << kAnsiReset
                      << error << "\n";
            return 1;
        }
        used_container = (parse_result == ContainerParseResult::kOk);

        if (used_container) {
            output_bytes_size = total_raw_bytes;
            {
                MANS_TIMING_SCOPE("alloc_output_buf");
                output_bytes.reset(static_cast<std::uint8_t*>(std::malloc(output_bytes_size)));
            }
            if (!output_bytes) {
                std::cerr << kAnsiRed << "Failed to allocate output buffer." << kAnsiReset
                          << "\n";
                return 1;
            }
            auto start = std::chrono::high_resolution_clock::now();
            std::size_t out_offset = 0;
            for (const auto& chunk : chunks) {
                std::size_t out_len = chunk.raw_len;
                mans::cpu::decompress_internal(
                    input_data.data() + chunk.payload_offset,
                    chunk.comp_len,
                    params,
                    output_bytes.get() + out_offset,
                    out_len,
                    save_adm,
                    output_file + ".adm");
                if (out_len != chunk.raw_len) {
                    std::cerr << kAnsiRed << "Chunk decompressed size mismatch." << kAnsiReset
                              << "\n";
                    return 1;
                }
                out_offset += out_len;
            }
            auto end = std::chrono::high_resolution_clock::now();
            decomp_ms = std::chrono::duration<double, std::milli>(end - start).count();
            output_write_size = output_bytes_size;
        } else {
            std::size_t estimated_out_size = input_data.size() * 10 + 4096;
            {
                MANS_TIMING_SCOPE("alloc_output_buf");
                output_bytes.reset(static_cast<std::uint8_t*>(std::malloc(estimated_out_size)));
            }
            if (!output_bytes) {
                std::cerr << kAnsiRed << "Failed to allocate output buffer." << kAnsiReset
                          << "\n";
                return 1;
            }
            std::size_t out_len = estimated_out_size;
            auto start = std::chrono::high_resolution_clock::now();
            mans::cpu::decompress_internal(
                input_data.data(),
                input_data.size(),
                params,
                output_bytes.get(),
                out_len,
                save_adm,
                output_file + ".adm");
            auto end = std::chrono::high_resolution_clock::now();
            decomp_ms = std::chrono::duration<double, std::milli>(end - start).count();

            if (out_len == 0) {
                std::cerr << kAnsiRed << "Decompression failed or returned 0 bytes."
                          << kAnsiReset << "\n";
                return 1;
            }
            if (out_len > estimated_out_size) {
                std::cerr << kAnsiYellow
                          << "[Warning] Decompressed size logic might be inconsistent."
                          << kAnsiReset << "\n";
            }
            output_bytes_size = out_len;
            chunk_count = 1;
            output_write_size = out_len;
        }

        io_write_ms = write_output_with_timing(output_file, output_bytes.get(), output_write_size);
        if (io_write_ms < 0.0) {
            std::cerr << kAnsiRed << "Failed to write output file: " << kAnsiReset
                      << output_file << "\n";
            return 1;
        }

        if (iter == 0) {
            continue;
        }
        total_decomp_ms += decomp_ms;
        total_io_read_ms += io_read_ms;
        total_io_write_ms += io_write_ms;
    }

    const double denom = static_cast<double>(kIters - 1);
    const double avg_decomp_ms = total_decomp_ms / denom;
    const double avg_io_read_ms = total_io_read_ms / denom;
    const double avg_io_write_ms = total_io_write_ms / denom;
    const double total_ms = avg_decomp_ms + avg_io_read_ms + avg_io_write_ms;
    const double throughput_mbps =
        (static_cast<double>(output_bytes_size) / 1e6) / (avg_decomp_ms / 1e3);
    const double io_ratio =
        total_ms > 0.0 ? (avg_io_read_ms + avg_io_write_ms) / total_ms : 0.0;

    std::cout << kAnsiBold << "Mans decompress finished!" << kAnsiReset
              << " Output: " << output_file << "\n";
    std::cout << kAnsiDim << "Config: " << kAnsiReset
              << "dtype=" << dtype_str
              << ", format=" << (used_container ? "container" : "legacy")
              << ", chunk_count=" << chunk_count
              << ", input_bytes=" << input_bytes
              << ", output_bytes=" << output_bytes_size
              << "\n";
    std::cout << kAnsiBlue << "Avg decomp ms" << kAnsiReset << ": "
              << std::fixed << std::setprecision(3) << avg_decomp_ms
              << ", " << kAnsiBlue << "Avg IO read ms" << kAnsiReset << ": "
              << avg_io_read_ms
              << ", " << kAnsiBlue << "Avg IO write ms" << kAnsiReset << ": "
              << avg_io_write_ms << "\n";
    std::cout << kAnsiGreen << "Throughput (MB/s, decomp only)" << kAnsiReset << ": "
              << std::fixed << std::setprecision(2) << throughput_mbps
              << ", " << kAnsiYellow << "IO ratio" << kAnsiReset << ": "
              << std::fixed << std::setprecision(4) << io_ratio
              << "\n";

    if (csv) {
        csv << input_bytes << ","
            << output_bytes_size << ","
            << chunk_count << ","
            << std::fixed << std::setprecision(3) << avg_decomp_ms << ","
            << std::fixed << std::setprecision(3) << avg_io_read_ms << ","
            << std::fixed << std::setprecision(3) << avg_io_write_ms << ","
            << std::fixed << std::setprecision(2) << throughput_mbps << ","
            << std::fixed << std::setprecision(4) << io_ratio << "\n";
    }

    MANS_TIMING_DUMP(output_file + ".timing.csv");

    return 0;
}
