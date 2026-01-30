// compiler: g++ -std=c++17 -O3 cpu_mans_compress.cpp mans_cpu.cpp -o cpu_mans_compress -fopenmp
// exec    : OMP_NUM_THREADS=4 ./cpu_mans_compress u2 input.u2 output.bin 1
//           OMP_NUM_THREADS=4 ./cpu_mans_compress u4 input.u4 output.bin 0

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "../mans_defs.h" 
#include "../mans_timing.h"
#include "adm/adm.h"
#include "adm/adm_utils.h"
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

struct ChunkInfo {
    std::size_t offset = 0;
    std::size_t len = 0;
};

template <typename T>
bool write_struct(std::vector<std::uint8_t>& data, std::size_t offset, const T& value) {
    if (offset + sizeof(T) > data.size()) {
        return false;
    }
    std::memcpy(data.data() + offset, &value, sizeof(T));
    return true;
}

std::vector<ChunkInfo> build_chunks(std::size_t total_elements, std::size_t chunk_elements) {
    std::vector<ChunkInfo> chunks;
    std::size_t offset = 0;
    while (offset < total_elements) {
        std::size_t len = std::min(chunk_elements, total_elements - offset);
        chunks.push_back(ChunkInfo{offset, len});
        offset += len;
    }
    return chunks;
}

std::vector<int> parse_threads(const std::string& arg, std::string& error) {
    std::vector<int> threads;
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            error = "Empty thread entry in --threads list.";
            return {};
        }
        std::size_t idx = 0;
        int value = 0;
        try {
            value = std::stoi(item, &idx);
        } catch (const std::exception&) {
            error = "Invalid thread value in --threads list: " + item;
            return {};
        }
        if (idx != item.size()) {
            error = "Invalid thread value in --threads list: " + item;
            return {};
        }
        if (value <= 0) {
            error = "Thread values must be positive integers.";
            return {};
        }
        threads.push_back(value);
    }
    return threads;
}

void apply_thread_overrides(mans::MansParams& params, const std::vector<int>& threads) {
    params.adm_decide_threads = threads[0];
    params.adm_center_calc_threads = threads[1];
    params.adm_encode_threads = threads[2];
    params.adm_warp_reduce_threads = threads[3];
    params.adm_fill_tail_threads = threads[4];
    params.adm_write_back_threads = threads[5];
    params.adm_restore_signals_threads = threads[6];
    params.adm_decode_values_threads = threads[7];
}

template <typename T>
bool load_input_with_timing(const std::string& input_file, std::vector<T>& host_data,
                            double& io_read_ms) {
    MANS_TIMING_SCOPE("io_read");
    auto start = std::chrono::high_resolution_clock::now();
    bool ok = false;
    if constexpr (std::is_same_v<T, std::uint16_t>) {
        ok = load_u16_file(input_file, host_data);
    } else {
        ok = load_u32_file(input_file, host_data);
    }
    auto end = std::chrono::high_resolution_clock::now();
    io_read_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return ok;
}

double write_output_with_timing(const std::string& output_file,
                                const std::vector<std::uint8_t>& data) {
    MANS_TIMING_SCOPE("io_write");
    auto start = std::chrono::high_resolution_clock::now();
    bool ok = save_u8_file(output_file, data);
    auto end = std::chrono::high_resolution_clock::now();
    if (!ok) {
        return -1.0;
    }
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename T>
bool compress_chunks(const std::vector<T>& host_data,
                     const mans::MansParams& params,
                     std::size_t chunk_elements,
                     bool save_adm,
                     const std::string& output_file,
                     std::vector<std::uint8_t>& compressed_data,
                     double& comp_ms) {
    std::vector<ChunkInfo> chunks = build_chunks(host_data.size(), chunk_elements);
    if (chunks.empty()) {
        return false;
    }

    std::size_t offset = 0;
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<std::uint8_t>> payloads;
    std::vector<std::uint64_t> raw_lens;
    payloads.reserve(chunks.size());
    raw_lens.reserve(chunks.size());
    std::size_t total_payload_bytes = 0;

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        const std::size_t input_bytes = chunk.len * sizeof(T);
        const std::size_t max_out_size = input_bytes * 2 + 4096;
        std::vector<std::uint8_t> chunk_payload(max_out_size);
        std::size_t out_size = chunk_payload.size();
        mans::cpu::compress_internal(
            host_data.data() + chunk.offset,
            chunk.len,
            params,
            chunk_payload.data(),
            out_size,
            save_adm,
            output_file + ".adm"
        );
        if (out_size == 0) {
            return false;
        }
        chunk_payload.resize(out_size);
        total_payload_bytes += sizeof(mans::container::ChunkHeader) + out_size;
        payloads.emplace_back(std::move(chunk_payload));
        raw_lens.push_back(static_cast<std::uint64_t>(input_bytes));
    }
    auto end = std::chrono::high_resolution_clock::now();
    comp_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    mans::container::ContainerHeader header{};
    std::memcpy(header.magic, mans::container::kContainerMagic,
                sizeof(mans::container::kContainerMagic));
    header.version = mans::container::kContainerVersion;
    header.dtype = (params.dtype == mans::DataType::U16) ? 1 : 2;
    header.reserved0 = 0;
    header.header_bytes = static_cast<std::uint16_t>(sizeof(mans::container::ContainerHeader));
    header.chunk_count = static_cast<std::uint64_t>(chunks.size());
    header.index_offset = sizeof(mans::container::ContainerHeader);
    header.data_offset = header.index_offset +
                         header.chunk_count * sizeof(mans::container::IndexEntry);
    header.chunk_header_bytes = sizeof(mans::container::ChunkHeader);
    header.flags = 0;

    const std::size_t total_size = static_cast<std::size_t>(header.data_offset) + total_payload_bytes;
    compressed_data.clear();
    compressed_data.resize(total_size);

    if (!write_struct(compressed_data, 0, header)) {
        return false;
    }
    for (std::size_t i = 0; i < payloads.size(); ++i) {
        mans::container::IndexEntry entry{};
        entry.offset = static_cast<std::uint64_t>(header.data_offset + offset);
        entry.comp_len = static_cast<std::uint64_t>(payloads[i].size());
        entry.raw_len = raw_lens[i];
        const std::size_t index_offset =
            static_cast<std::size_t>(header.index_offset + i * sizeof(mans::container::IndexEntry));
        if (!write_struct(compressed_data, index_offset, entry)) {
            return false;
        }
        mans::container::ChunkHeader chunk_header{};
        std::memcpy(chunk_header.magic, mans::container::kChunkMagic,
                    sizeof(mans::container::kChunkMagic));
        chunk_header.version = mans::container::kChunkVersion;
        chunk_header.header_bytes = static_cast<std::uint16_t>(sizeof(mans::container::ChunkHeader));
        chunk_header.comp_len = entry.comp_len;
        chunk_header.raw_len = entry.raw_len;
        chunk_header.chunk_index = static_cast<std::uint64_t>(i);
        if (!write_struct(compressed_data, static_cast<std::size_t>(entry.offset), chunk_header)) {
            return false;
        }
        const std::size_t payload_offset =
            static_cast<std::size_t>(entry.offset + sizeof(mans::container::ChunkHeader));
        if (payload_offset + payloads[i].size() > compressed_data.size()) {
            return false;
        }
        std::memcpy(compressed_data.data() + payload_offset,
                    payloads[i].data(),
                    payloads[i].size());
        offset += sizeof(mans::container::ChunkHeader) + payloads[i].size();
    }

    return offset == total_payload_bytes;
}

} // namespace

int main(int argc, char** argv) {

    if (argc < 5) {
        std::cerr << kAnsiRed << "Use: " << kAnsiReset << argv[0] 
                  << " <u2|u4> <input_file> <output_bin_file> <save_adm(0|1)>"
                  << " [--threshold 4000] [--chunk-mb 0.0]"
                  << " [--threads 16,32,32,32,32,32,32,32]"
                  << " [--csv out.csv]\n";
        return 1;
    }

    std::string dtype_str   = argv[1];
    std::string input_file  = argv[2];
    std::string output_file = argv[3];
    std::string save_flag   = argv[4];
    
    // 1. save intermediate ADM compressed data or not
    bool save_adm = (save_flag == "1");
    uint32_t threshold = 4000;
    double chunk_mb = 0.0;
    std::string csv_path;
    std::string threads_arg;
    bool has_threads = false;

    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threshold" && i + 1 < argc) {
            threshold = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--chunk-mb" && i + 1 < argc) {
            chunk_mb = std::stod(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads_arg = argv[++i];
            has_threads = true;
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        }
    }

    // 2. build MansParams
    mans::MansParams params{};
    params.backend = mans::Backend::CPU;
    params.adm_threshold = threshold;
    params.adm_decide_threads = 16;
    params.adm_center_calc_threads = 32;
    params.adm_encode_threads = 32;
    params.adm_warp_reduce_threads = 32;
    params.adm_fill_tail_threads = 32;
    params.adm_write_back_threads = 32;
    params.adm_restore_signals_threads = 32;
    params.adm_decode_values_threads = 32;

    if (dtype_str == "u2" || dtype_str == "-u2") {
        params.dtype = mans::DataType::U16;
    } else if (dtype_str == "u4" || dtype_str == "-u4") {
        params.dtype = mans::DataType::U32;
    } else {
        std::cerr << "Unknown data type flag: " << dtype_str << "\nUse: u2 or u4\n";
        return 1;
    }

    if (has_threads) {
        std::string error;
        std::vector<int> thread_list = parse_threads(threads_arg, error);
        if (!error.empty()) {
            std::cerr << error << "\n";
            return 1;
        }
        if (thread_list.size() != 8) {
            std::cerr << "--threads expects 8 values: decide,center,encode,warp_reduce,"
                      << "fill_tail,write_back,restore_signals,decode_values\n";
            return 1;
        }
        apply_thread_overrides(params, thread_list);
    }

    std::cout << kAnsiBold << "Command-line arguments:" << kAnsiReset << "\n";
    std::cout << "  " << kAnsiCyan << "Input type" << kAnsiReset << ": " << dtype_str << "\n";
    std::cout << "  " << kAnsiCyan << "Input file" << kAnsiReset << ": " << input_file << "\n";
    std::cout << "  " << kAnsiCyan << "Output file" << kAnsiReset << ": " << output_file << "\n";
    std::cout << "  " << kAnsiCyan << "Save ADM" << kAnsiReset << ": "
              << (save_adm ? kAnsiGreen : kAnsiYellow)
              << (save_adm ? "yes" : "no") << kAnsiReset << "\n";
    std::cout << "  " << kAnsiCyan << "Threshold" << kAnsiReset << ": " << threshold << "\n";
    if (chunk_mb > 0.0) {
        std::cout << "  " << kAnsiCyan << "Chunk size (MB)" << kAnsiReset << ": " << chunk_mb << "\n";
    } else {
        std::cout << "  " << kAnsiCyan << "Chunk size (MB)" << kAnsiReset << ": full input\n";
    }
    if (has_threads) {
        std::cout << "  " << kAnsiCyan << "Threads" << kAnsiReset << ": " << threads_arg << "\n";
    }
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
        csv << "input_bytes,chunk_bytes,comp_ms,io_read_ms,io_write_ms,"
               "throughput_mbps,io_ratio\n";
    }

    constexpr int kIters = 11;
    double total_comp_ms = 0.0;
    double total_io_read_ms = 0.0;
    double total_io_write_ms = 0.0;
    std::size_t input_bytes = 0;
    std::size_t chunk_bytes = 0;

    for (int iter = 0; iter < kIters; ++iter) {
        MANS_TIMING_RUN_SCOPE();
        MANS_TIMING_SCOPE("total");
        std::vector<uint8_t> compressed_data;
        double io_read_ms = 0.0;
        double io_write_ms = 0.0;
        double comp_ms = 0.0;
        bool ok = false;

        if (params.dtype == mans::DataType::U16) {
            std::vector<uint16_t> host_data;
            if (!load_input_with_timing(input_file, host_data, io_read_ms)) {
                std::cerr << kAnsiRed << "Failed to load input file: " << kAnsiReset
                          << input_file << "\n";
                return 1;
            }
            if (host_data.empty()) {
                std::cerr << kAnsiRed << "Input file is empty." << kAnsiReset << "\n";
                return 1;
            }
            input_bytes = host_data.size() * sizeof(uint16_t);

            std::size_t chunk_elements_local = 0;
            if (chunk_mb > 0.0) {
                double chunk_bytes_double = chunk_mb * 1024.0 * 1024.0;
                chunk_elements_local =
                    static_cast<std::size_t>(chunk_bytes_double / sizeof(uint16_t));
            }
            if (chunk_elements_local == 0) {
                chunk_elements_local = host_data.size();
            }
            chunk_bytes = chunk_elements_local * sizeof(uint16_t);

            ok = compress_chunks<uint16_t>(host_data, params, chunk_elements_local,
                                           save_adm, output_file,
                                           compressed_data, comp_ms);
        } else { // U32
            std::vector<uint32_t> host_data;
            if (!load_input_with_timing(input_file, host_data, io_read_ms)) {
                std::cerr << kAnsiRed << "Failed to load input file: " << kAnsiReset
                          << input_file << "\n";
                return 1;
            }
            if (host_data.empty()) {
                std::cerr << kAnsiRed << "Input file is empty." << kAnsiReset << "\n";
                return 1;
            }
            input_bytes = host_data.size() * sizeof(uint32_t);

            std::size_t chunk_elements_local = 0;
            if (chunk_mb > 0.0) {
                double chunk_bytes_double = chunk_mb * 1024.0 * 1024.0;
                chunk_elements_local =
                    static_cast<std::size_t>(chunk_bytes_double / sizeof(uint32_t));
            }
            if (chunk_elements_local == 0) {
                chunk_elements_local = host_data.size();
            }
            chunk_bytes = chunk_elements_local * sizeof(uint32_t);

            ok = compress_chunks<uint32_t>(host_data, params, chunk_elements_local,
                                           save_adm, output_file,
                                           compressed_data, comp_ms);
        }

        if (!ok) {
            std::cerr << kAnsiRed << "Compression failed." << kAnsiReset << "\n";
            return 1;
        }

        io_write_ms = write_output_with_timing(output_file, compressed_data);
        if (io_write_ms < 0.0) {
            std::cerr << kAnsiRed << "Failed to write Final output: " << kAnsiReset
                      << output_file << "\n";
            return 1;
        }

        if (iter == 0) {
            continue;
        }
        total_comp_ms += comp_ms;
        total_io_read_ms += io_read_ms;
        total_io_write_ms += io_write_ms;
    }

    const double denom = static_cast<double>(kIters - 1);
    const double avg_comp_ms = total_comp_ms / denom;
    const double avg_io_read_ms = total_io_read_ms / denom;
    const double avg_io_write_ms = total_io_write_ms / denom;
    const double total_ms = avg_comp_ms + avg_io_read_ms + avg_io_write_ms;
    const double throughput_mbps =
        (static_cast<double>(input_bytes) / 1e6) / (avg_comp_ms / 1e3);
    const double io_ratio =
        total_ms > 0.0 ? (avg_io_read_ms + avg_io_write_ms) / total_ms : 0.0;

    std::cout << kAnsiBold << "Mans compress finished!" << kAnsiReset
              << " Output: " << output_file << "\n";
    std::cout << kAnsiDim << "Config: " << kAnsiReset
              << "dtype=" << dtype_str
              << ", threshold=" << threshold
              << ", chunk_bytes=" << chunk_bytes
              << "\n";
    std::cout << kAnsiBlue << "Avg comp ms" << kAnsiReset << ": "
              << std::fixed << std::setprecision(3) << avg_comp_ms
              << ", " << kAnsiBlue << "Avg IO read ms" << kAnsiReset << ": "
              << avg_io_read_ms
              << ", " << kAnsiBlue << "Avg IO write ms" << kAnsiReset << ": "
              << avg_io_write_ms << "\n";
    std::cout << kAnsiGreen << "Throughput (MB/s, comp only)" << kAnsiReset << ": "
              << std::fixed << std::setprecision(2) << throughput_mbps
              << ", " << kAnsiYellow << "IO ratio" << kAnsiReset << ": "
              << std::fixed << std::setprecision(4) << io_ratio
              << "\n";

    if (csv) {
        csv << input_bytes << ","
            << chunk_bytes << ","
            << std::fixed << std::setprecision(3) << avg_comp_ms << ","
            << std::fixed << std::setprecision(3) << avg_io_read_ms << ","
            << std::fixed << std::setprecision(3) << avg_io_write_ms << ","
            << std::fixed << std::setprecision(2) << throughput_mbps << ","
            << std::fixed << std::setprecision(4) << io_ratio << "\n";
    }

    MANS_TIMING_DUMP(output_file + ".timing.csv");

    return 0;
}
