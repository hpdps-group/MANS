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

#include "../mans_defs.h" 
#include "../mans_timing.h"
#include "adm/adm.h"
#include "adm/adm_utils.h"
#include "mans_cpu.h"
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
    params.adm_center_calc_threads = threads[0];
    params.adm_encode_threads = threads[1];
    params.adm_warp_reduce_threads = threads[2];
    params.adm_fill_tail_threads = threads[3];
    params.adm_write_back_threads = threads[4];
    params.adm_restore_signals_threads = threads[5];
    params.adm_decode_values_threads = threads[6];
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

    std::size_t max_out_size = 0;
    std::size_t max_chunk_len = 0;
    for (const auto& chunk : chunks) {
        std::size_t input_bytes = chunk.len * sizeof(T);
        max_out_size += input_bytes * 2 + 4096;
        if (chunk.len > max_chunk_len) {
            max_chunk_len = chunk.len;
        }
    }
    compressed_data.clear();
    compressed_data.resize(max_out_size);
    
    std::unique_ptr<std::uint8_t, decltype(&free)> mans_intermediate_buf(nullptr, &free);
    std::size_t mans_intermediate_cap = 0;
    if (max_chunk_len > 0) {
        MANS_TIMING_SCOPE("alloc_mans_intermediate_buf");
        mans_intermediate_cap = adm_max_compressed_size<T>(max_chunk_len);
        mans_intermediate_buf.reset(
            static_cast<std::uint8_t*>(std::malloc(mans_intermediate_cap)));
    }

    std::size_t offset = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        std::size_t out_size = max_out_size - offset;
        mans::cpu::compress_internal(
            host_data.data() + chunk.offset,
            chunk.len,
            params,
            compressed_data.data() + offset,
            out_size,
            save_adm,
            output_file + ".adm",
            mans_intermediate_buf.get(),
            mans_intermediate_cap
        );
        offset += out_size;
    }
    auto end = std::chrono::high_resolution_clock::now();
    comp_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    if (offset > compressed_data.size()) {
        return false;
    }
    compressed_data.resize(offset);
    return true;
}

} // namespace

int main(int argc, char** argv) {

    if (argc < 5) {
        std::cerr << kAnsiRed << "Use: " << kAnsiReset << argv[0] 
                  << " <u2|u4> <input_file> <output_bin_file> <save_adm(0|1)>"
                  << " [--threshold 4000] [--chunk-mb 0.0]"
                  << " [--threads 32,32,32,32,32,32,32]"
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
        if (thread_list.size() != 7) {
            std::cerr << "--threads expects 7 values: center,encode,warp_reduce,"
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

    constexpr int kIters = 2;
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
