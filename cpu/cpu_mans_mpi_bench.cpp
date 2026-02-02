// compiler: mpicxx -std=c++17 -O3 cpu_mans_mpi_bench.cpp -o cpu_mans_mpi_bench
// exec    : mpirun -n 4 ./cpu_mans_mpi_bench --config /path/to/mpi_mans.cfg

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

#include "../mans_api.hpp"
#include "../mans_data_gen.h"
#include "../mans_defs.h"
#include "../mans_timing.h"
#include "adm/adm_utils.h"
#include "mans_cpu.h"
#include "mans_container.h"
#include "../tools/H5Z-MANS/include/H5Z-MANS_config.h"

namespace {

constexpr double kDefaultChunkMb = 32.0;
constexpr const char* kAnsiReset = "\033[0m";
constexpr const char* kAnsiBold = "\033[1m";
constexpr const char* kAnsiDim = "\033[2m";
constexpr const char* kAnsiRed = "\033[31m";
constexpr const char* kAnsiGreen = "\033[32m";
constexpr const char* kAnsiYellow = "\033[33m";
constexpr const char* kAnsiBlue = "\033[34m";
constexpr const char* kAnsiCyan = "\033[36m";

enum class FilterId : std::uint32_t {
    None = 0,
    Mans = 1,
};

enum class BenchMode : std::uint32_t {
    Bench = 0,
    Compress = 1,
    Decompress = 2,
};

static bool parse_mode(std::string_view text, BenchMode& mode) {
    if (text == "bench") {
        mode = BenchMode::Bench;
        return true;
    }
    if (text == "compress") {
        mode = BenchMode::Compress;
        return true;
    }
    if (text == "decompress") {
        mode = BenchMode::Decompress;
        return true;
    }
    return false;
}

static const char* mode_to_string(BenchMode mode) {
    switch (mode) {
        case BenchMode::Bench:
            return "bench";
        case BenchMode::Compress:
            return "compress";
        case BenchMode::Decompress:
            return "decompress";
    }
    return "bench";
}

struct RunOptions {
    std::string input_bin;
    std::string output_bin;
    std::string mans_config_file;
    std::string dataset_config_file;
    std::string metrics_csv;

    double chunk_mb = kDefaultChunkMb;
    FilterId filter = FilterId::Mans;
    BenchMode mode = BenchMode::Bench;
    int expected_ranks = -1;
    std::optional<std::uint32_t> dtype_override;
    bool filter_seen = false;
    bool mode_seen = false;
    bool raw_io = false;
};

struct Metrics {
    double io_read_s = 0.0;
    double comp_s = 0.0;
    double io_write_s = 0.0;
    std::uint64_t raw_bytes = 0;
    std::uint64_t comp_bytes = 0;
};


struct MpiContext {
    int rank = 0;
    int ranks = 1;
    bool initialized = false;

    MpiContext(int& argc, char**& argv) {
        int is_init = 0;
        MPI_Initialized(&is_init);
        if (!is_init) {
            MPI_Init(&argc, &argv);
            initialized = true;
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    }

    ~MpiContext() {
        int is_finalized = 0;
        MPI_Finalized(&is_finalized);
        if (initialized && !is_finalized) {
            MPI_Finalize();
        }
    }
};

struct ChunkInfo {
    std::size_t offset = 0;
    std::size_t len = 0;
};

struct PayloadBlock {
    std::unique_ptr<std::uint8_t, decltype(&std::free)> data{nullptr, &std::free};
    std::size_t size = 0;
};

static std::vector<ChunkInfo> build_chunks(std::size_t total_elements, std::size_t chunk_elements) {
    std::vector<ChunkInfo> chunks;
    if (chunk_elements == 0) {
        return chunks;
    }
    std::size_t offset = 0;
    while (offset < total_elements) {
        const std::size_t len = std::min(chunk_elements, total_elements - offset);
        chunks.push_back(ChunkInfo{offset, len});
        offset += len;
    }
    return chunks;
}

static std::string trim_copy(const std::string& str) {
    const char* whitespace = " \t\r\n";
    const auto first = str.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        return "";
    }
    const auto last = str.find_last_not_of(whitespace);
    return str.substr(first, last - first + 1);
}

static FilterId parse_filter(std::string_view name) {
    if (name == "none") {
        return FilterId::None;
    }
    if (name == "mans") {
        return FilterId::Mans;
    }
    throw std::runtime_error("Unknown filter: " + std::string(name));
}

static bool parse_bool(std::string_view val, bool& out) {
    if (val == "1" || val == "true" || val == "True" || val == "yes" || val == "on") {
        out = true;
        return true;
    }
    if (val == "0" || val == "false" || val == "False" || val == "no" || val == "off") {
        out = false;
        return true;
    }
    return false;
}

static void split_even(std::size_t total, int rank, int ranks, std::size_t& offset, std::size_t& count) {
    const std::size_t base = total / static_cast<std::size_t>(ranks);
    const std::size_t rem = total % static_cast<std::size_t>(ranks);
    if (static_cast<std::size_t>(rank) < rem) {
        count = base + 1;
        offset = static_cast<std::size_t>(rank) * count;
    } else {
        count = base;
        offset = rem * (base + 1) + (static_cast<std::size_t>(rank) - rem) * base;
    }
}

static std::uint64_t file_size_bytes(const std::string& path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("Failed to stat file: " + path);
    }
    return static_cast<std::uint64_t>(st.st_size);
}

static void read_fully(int fd, void* buf, std::size_t bytes, std::uint64_t offset) {
    std::size_t done = 0;
    while (done < bytes) {
        const auto rc = pread(fd,
                              static_cast<char*>(buf) + done,
                              bytes - done,
                              static_cast<off_t>(offset + done));
        if (rc < 0) {
            throw std::runtime_error("pread failed");
        }
        if (rc == 0) {
            throw std::runtime_error("Unexpected EOF during pread");
        }
        done += static_cast<std::size_t>(rc);
    }
}

static void write_fully(int fd, const void* buf, std::size_t bytes, std::uint64_t offset) {
    std::size_t done = 0;
    while (done < bytes) {
        const auto rc = pwrite(fd,
                               static_cast<const char*>(buf) + done,
                               bytes - done,
                               static_cast<off_t>(offset + done));
        if (rc < 0) {
            throw std::runtime_error("pwrite failed");
        }
        if (rc == 0) {
            throw std::runtime_error("Short pwrite");
        }
        done += static_cast<std::size_t>(rc);
    }
}

static std::optional<RunOptions> parse_args(int argc, char** argv, int rank) {
    RunOptions opts;
    std::string config_file;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --config");
            }
            config_file = argv[++i];
        }
    }

    if (!config_file.empty()) {
        std::ifstream in(config_file);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open config: " + config_file);
        }
        std::string line;
        while (std::getline(in, line)) {
            const auto hash = line.find('#');
            if (hash != std::string::npos) {
                line = line.substr(0, hash);
            }
            line = trim_copy(line);
            if (line.empty()) {
                continue;
            }
            const auto eq = line.find('=');
            if (eq == std::string::npos) {
                continue;
            }
            const auto key = trim_copy(line.substr(0, eq));
            const auto val = trim_copy(line.substr(eq + 1));
            if (key.empty() || val.empty()) {
                continue;
            }
            if (key == "input" || key == "input_bin") {
                opts.input_bin = val;
                continue;
            }
            if (key == "output" || key == "output_bin" || key == "output_h5") {
                opts.output_bin = val;
                continue;
            }
            if (key == "chunk" || key == "chunk_mb") {
                opts.chunk_mb = std::stod(val);
                continue;
            }
            if (key == "filter") {
                opts.filter = parse_filter(val);
                opts.filter_seen = true;
                continue;
            }
            if (key == "mode") {
                opts.mode_seen = true;
                if (!parse_mode(val, opts.mode)) {
                    throw std::runtime_error(
                        "mode must be one of 'compress', 'decompress', 'bench' for cpu_mans_mpi_bench");
                }
                continue;
            }
            if (key == "ranks" || key == "expected_ranks") {
                opts.expected_ranks = std::stoi(val);
                continue;
            }
            if (key == "mans_config" || key == "mans_config_file") {
                opts.mans_config_file = val;
                continue;
            }
            if (key == "dataset_config" || key == "dataset_config_file") {
                opts.dataset_config_file = val;
                continue;
            }
            if (key == "csv" || key == "metrics_csv") {
                opts.metrics_csv = val;
                continue;
            }
            if (key == "dtype") {
                opts.dtype_override = mans::data_gen::parse_dtype_value(val);
                continue;
            }
            if (key == "raw_io") {
                bool value = false;
                if (!parse_bool(val, value)) {
                    throw std::runtime_error("Invalid raw_io value: " + val);
                }
                opts.raw_io = value;
                continue;
            }
            if (rank == 0) {
                std::cerr << "[WARN] Unknown config key: " << key << "\n";
            }
        }
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--config") {
            (void)need_value("--config");
            continue;
        }
        if (arg == "--input") {
            opts.input_bin = need_value("--input");
            continue;
        }
        if (arg == "--output") {
            opts.output_bin = need_value("--output");
            continue;
        }
        if (arg == "--chunk-mb") {
            opts.chunk_mb = std::stod(need_value("--chunk-mb"));
            continue;
        }
        if (arg == "--filter") {
            opts.filter = parse_filter(need_value("--filter"));
            opts.filter_seen = true;
            continue;
        }
        if (arg == "--mode") {
            opts.mode_seen = true;
            const auto mode_val = need_value("--mode");
            if (!parse_mode(mode_val, opts.mode)) {
                throw std::runtime_error(
                    "mode must be one of 'compress', 'decompress', 'bench' for cpu_mans_mpi_bench");
            }
            continue;
        }
        if (arg == "--ranks") {
            opts.expected_ranks = std::stoi(need_value("--ranks"));
            continue;
        }
        if (arg == "--mans-config") {
            opts.mans_config_file = need_value("--mans-config");
            continue;
        }
        if (arg == "--dataset-config") {
            opts.dataset_config_file = need_value("--dataset-config");
            continue;
        }
        if (arg == "--csv") {
            opts.metrics_csv = need_value("--csv");
            continue;
        }
        if (arg == "--dtype") {
            opts.dtype_override = mans::data_gen::parse_dtype_value(need_value("--dtype"));
            continue;
        }
        if (arg == "--raw-io") {
            opts.raw_io = true;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            if (rank == 0) {
                std::cerr << "Usage:\n  " << argv[0]
                          << " --config mpi_mans.cfg [--input raw.bin] [--output out.bin]\\\n"
                             "         [--chunk-mb MB] [--filter mans|none] [--mode compress|decompress|bench]\\\n"
                             "         [--mans-config mans.cfg]\\\n"
                             "         [--dataset-config synth.cfg] [--dtype u16|u32] [--csv metrics.csv]\\\n"
                             "         [--raw-io]\n";
            }
            return std::nullopt;
        }

        throw std::runtime_error("Unknown argument: " + arg);
    }

    if (opts.mode == BenchMode::Decompress && (opts.input_bin.empty() || opts.output_bin.empty())) {
        if (rank == 0) {
            std::cerr << "[Error] --mode decompress requires both --input and --output.\n";
        }
        return std::nullopt;
    }

    if (opts.mode != BenchMode::Decompress && opts.output_bin.empty() && opts.dataset_config_file.empty()) {
        if (rank == 0) {
            std::cerr << "[Error] output_bin is required (or provide dataset_config_file with output_bin).\n";
        }
        return std::nullopt;
    }

    if (opts.mode != BenchMode::Decompress &&
        opts.filter == FilterId::Mans &&
        opts.mans_config_file.empty()) {
        if (rank == 0) {
            std::cerr << "[Error] --mans-config is required when --filter mans\n";
        }
        return std::nullopt;
    }

    return opts;
}

static void append_metrics_csv(const std::string& csv_path,
                               const RunOptions& opts,
                               std::size_t total_elems,
                               std::size_t elem_size,
                               int ranks,
                               const Metrics& agg) {
    std::ofstream out(csv_path, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "[WARN] Failed to open metrics CSV: " << csv_path << "\n";
        return;
    }

    const bool is_new = out.tellp() == 0;
    if (is_new) {
        out << "mode,filter,chunk_mb,ranks,total_elems,elem_size_bytes,raw_bytes,comp_bytes,"
               "io_read_s,comp_s,io_write_s,read_thr_mb_s,comp_thr_mb_s,write_thr_mb_s,comp_write_thr_mb_s,ratio\n";
    }

    const double raw_mb = static_cast<double>(agg.raw_bytes) / 1048576.0;
    const double comp_mb = static_cast<double>(agg.comp_bytes) / 1048576.0;
    const double read_thr = agg.io_read_s > 0.0
        ? ((opts.mode == BenchMode::Decompress) ? (comp_mb / agg.io_read_s) : (raw_mb / agg.io_read_s))
        : 0.0;
    const double comp_thr = agg.comp_s > 0.0 ? raw_mb / agg.comp_s : 0.0;
    const double write_thr = agg.io_write_s > 0.0 ? raw_mb / agg.io_write_s : 0.0;
    const double comp_write_s = agg.comp_s + agg.io_write_s;
    const double comp_write_thr = comp_write_s > 0.0 ? raw_mb / comp_write_s : 0.0;
    const double ratio = agg.raw_bytes > 0 ? static_cast<double>(agg.comp_bytes) / agg.raw_bytes : 0.0;

    out << mode_to_string(opts.mode) << ","
        << (opts.filter == FilterId::Mans ? "mans" : "none") << ","
        << (opts.mode == BenchMode::Decompress ? 0.0 : opts.chunk_mb) << ","
        << ranks << ","
        << total_elems << ","
        << elem_size << ","
        << agg.raw_bytes << ","
        << agg.comp_bytes << ","
        << agg.io_read_s << ","
        << agg.comp_s << ","
        << agg.io_write_s << ","
        << read_thr << ","
        << comp_thr << ","
        << write_thr << ","
        << comp_write_thr << ","
        << ratio
        << "\n";
}

static int run_decompress_mode(const MpiContext& mpi, RunOptions opts) {
    mans::container::ContainerHeader header{};
    std::vector<mans::container::IndexEntry> index_entries;
    if (mpi.rank == 0) {
        int fd = open(opts.input_bin.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "[Error] Failed to open input_bin: " << opts.input_bin << "\n";
            return 1;
        }
        read_fully(fd, &header, sizeof(header), 0);
        if (!mans::container::magic_matches(
                header.magic, mans::container::kContainerMagic,
                sizeof(mans::container::kContainerMagic)) ||
            header.version != mans::container::kContainerVersion ||
            header.header_bytes != sizeof(mans::container::ContainerHeader)) {
            std::cerr << "[Error] Invalid container header (magic/version mismatch)\n";
            close(fd);
            return 1;
        }
        if (header.chunk_header_bytes != sizeof(mans::container::ChunkHeader)) {
            std::cerr << "[Error] Chunk header size mismatch\n";
            close(fd);
            return 1;
        }
        index_entries.resize(static_cast<std::size_t>(header.chunk_count));
        if (header.chunk_count > 0) {
            read_fully(fd,
                       index_entries.data(),
                       index_entries.size() * sizeof(mans::container::IndexEntry),
                       static_cast<std::uint64_t>(header.index_offset));
        }
        close(fd);
    }

    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (mpi.rank != 0) {
        index_entries.resize(static_cast<std::size_t>(header.chunk_count));
    }
    if (!index_entries.empty()) {
        MPI_Bcast(index_entries.data(),
                  static_cast<int>(index_entries.size() * sizeof(mans::container::IndexEntry)),
                  MPI_BYTE,
                  0,
                  MPI_COMM_WORLD);
    }

    const bool raw_payload =
        (header.flags & mans::container::kFlagRawPayload) != 0;
    const auto header_filter = raw_payload ? FilterId::None : FilterId::Mans;
    if (!opts.filter_seen) {
        opts.filter = header_filter;
    } else if (opts.filter != header_filter) {
        if (mpi.rank == 0) {
            std::cerr << "[Error] Filter mismatch: config/CLI says "
                      << (opts.filter == FilterId::Mans ? "mans" : "none")
                      << ", header says " << (header_filter == FilterId::Mans ? "mans" : "none")
                      << "\n";
        }
        return 1;
    }

    if (opts.filter == FilterId::Mans && opts.mans_config_file.empty()) {
        if (mpi.rank == 0) {
            std::cerr << "[Error] --mans-config is required when filter is mans\n";
        }
        return 1;
    }

    std::optional<mans::h5::MansConfig> mans_cfg;
    mans::MansParams params{};
    params.backend = mans::Backend::CPU;
    params.dtype = mans::DataType::U32;
    params.adm_threshold = 4000U;

    if (opts.filter == FilterId::Mans) {
        mans_cfg.emplace();
        try {
            mans_cfg->load(opts.mans_config_file);
        } catch (const std::exception& e) {
            if (mpi.rank == 0) {
                std::cerr << "MANS config error: " << e.what() << "\n";
            }
            return 1;
        }
        params = mans_cfg->get_params();
        params.backend = mans::Backend::CPU;
    }

    if (header.dtype != 1 && header.dtype != 2) {
        if (mpi.rank == 0) {
            std::cerr << "[Error] Unsupported dtype value in container header: "
                      << static_cast<int>(header.dtype) << "\n";
        }
        return 1;
    }
    params.dtype = (header.dtype == 1) ? mans::DataType::U16 : mans::DataType::U32;
    const std::size_t elem_size = (params.dtype == mans::DataType::U16) ? 2 : 4;
    std::uint64_t total_raw_bytes = 0;
    for (const auto& entry : index_entries) {
        total_raw_bytes += entry.raw_len;
    }
    if (total_raw_bytes % elem_size != 0) {
        if (mpi.rank == 0) {
            std::cerr << "[Error] Total raw bytes not divisible by elem size.\n";
        }
        return 1;
    }
    const std::size_t total_elems = static_cast<std::size_t>(total_raw_bytes / elem_size);

    if (opts.expected_ranks > 0 && opts.expected_ranks != mpi.ranks && mpi.rank == 0) {
        std::cerr << "[WARN] --ranks " << opts.expected_ranks
                  << " does not match MPI world size " << mpi.ranks << ". Using MPI size.\n";
    }

    std::size_t rank_offset = 0;
    std::size_t rank_count = 0;
    split_even(total_elems, mpi.rank, mpi.ranks, rank_offset, rank_count);

    if (mpi.rank == 0) {
        const std::string dtype_str = (params.dtype == mans::DataType::U16) ? "u2" : "u4";
        std::cout << "  " << kAnsiCyan << "OMP max threads" << kAnsiReset << ": "
                  << omp_get_max_threads() << "\n";
        std::cout << kAnsiBold << "Command-line arguments:" << kAnsiReset << "\n";
        std::cout << "  " << kAnsiCyan << "Mode" << kAnsiReset << ": " << mode_to_string(opts.mode) << "\n";
        std::cout << "  " << kAnsiCyan << "Input type" << kAnsiReset << ": " << dtype_str << "\n";
        std::cout << "  " << kAnsiCyan << "Input file" << kAnsiReset << ": " << opts.input_bin << "\n";
        std::cout << "  " << kAnsiCyan << "Output file" << kAnsiReset << ": " << opts.output_bin << "\n";
        std::cout << "  " << kAnsiCyan << "Filter" << kAnsiReset << ": "
                  << (opts.filter == FilterId::Mans ? "mans" : "none") << "\n";
        std::cout << "  " << kAnsiCyan << "MPI ranks" << kAnsiReset << ": " << mpi.ranks << "\n";
        if (!opts.metrics_csv.empty()) {
            std::cout << "  " << kAnsiCyan << "CSV output" << kAnsiReset << ": " << opts.metrics_csv << "\n";
        }
        std::cout << "\n";
    }

    if (mpi.rank == 0) {
        int fd = open(opts.output_bin.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (fd < 0) {
            std::cerr << "[Error] Failed to create output_bin: " << opts.output_bin << "\n";
            return 1;
        }
        if (ftruncate(fd, static_cast<off_t>(total_raw_bytes)) != 0) {
            std::cerr << "[WARN] Failed to ftruncate output file.\n";
        }
        close(fd);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    Metrics local{};
    local.raw_bytes = static_cast<std::uint64_t>(rank_count) * elem_size;
    local.comp_bytes = 0;

    if (rank_count > 0 && total_raw_bytes > 0) {
        int in_fd = open(opts.input_bin.c_str(), O_RDONLY);
        int out_fd = open(opts.output_bin.c_str(), O_WRONLY);
        if (in_fd < 0 || out_fd < 0) {
            if (mpi.rank == 0) {
                std::cerr << "[Error] Failed to open input/output files.\n";
            }
            if (in_fd >= 0) {
                close(in_fd);
            }
            if (out_fd >= 0) {
                close(out_fd);
            }
            return 1;
        }

        const std::size_t raw_bytes = rank_count * elem_size;
        std::vector<std::uint8_t> raw_buf(raw_bytes, 0);
        const std::uint64_t rank_start = static_cast<std::uint64_t>(rank_offset) * elem_size;
        const std::uint64_t rank_end = rank_start + raw_bytes;

        std::uint64_t chunk_raw_start = 0;
        for (std::size_t i = 0; i < index_entries.size(); ++i) {
            const auto& entry = index_entries[i];
            const std::uint64_t chunk_raw_end = chunk_raw_start + entry.raw_len;
            if (chunk_raw_end <= rank_start || chunk_raw_start >= rank_end) {
                chunk_raw_start = chunk_raw_end;
                continue;
            }

            mans::container::ChunkHeader chunk_header{};
            read_fully(in_fd, &chunk_header, sizeof(chunk_header), entry.offset);
            if (!mans::container::magic_matches(
                    chunk_header.magic, mans::container::kChunkMagic,
                    sizeof(mans::container::kChunkMagic)) ||
                chunk_header.version != mans::container::kChunkVersion ||
                chunk_header.header_bytes != sizeof(mans::container::ChunkHeader) ||
                chunk_header.comp_len != entry.comp_len ||
                chunk_header.raw_len != entry.raw_len) {
                throw std::runtime_error("Invalid chunk header");
            }

            const std::uint64_t payload_offset =
                entry.offset + sizeof(mans::container::ChunkHeader);

            const std::uint64_t copy_start = std::max(chunk_raw_start, rank_start);
            const std::uint64_t copy_end = std::min(chunk_raw_end, rank_end);
            const std::size_t copy_len = static_cast<std::size_t>(copy_end - copy_start);
            const std::size_t chunk_offset = static_cast<std::size_t>(copy_start - chunk_raw_start);
            const std::size_t rank_offset_bytes = static_cast<std::size_t>(copy_start - rank_start);

            if (chunk_raw_start >= rank_start && chunk_raw_start < rank_end) {
                local.comp_bytes += entry.comp_len;
            }

            if (raw_payload) {
                auto t0 = std::chrono::steady_clock::now();
                read_fully(in_fd,
                           raw_buf.data() + rank_offset_bytes,
                           copy_len,
                           payload_offset + chunk_offset);
                auto t1 = std::chrono::steady_clock::now();
                local.io_read_s += std::chrono::duration<double>(t1 - t0).count();
            } else {
                std::vector<std::uint8_t> comp_buf(static_cast<std::size_t>(entry.comp_len));
                auto t0 = std::chrono::steady_clock::now();
                read_fully(in_fd, comp_buf.data(), comp_buf.size(), payload_offset);
                auto t1 = std::chrono::steady_clock::now();
                local.io_read_s += std::chrono::duration<double>(t1 - t0).count();

                std::vector<std::uint8_t> chunk_raw(static_cast<std::size_t>(entry.raw_len));
                std::size_t out_size = chunk_raw.size();
                auto t2 = std::chrono::steady_clock::now();
                mans::decompress(comp_buf.data(),
                                 comp_buf.size(),
                                 params,
                                 chunk_raw.data(),
                                 out_size);
                auto t3 = std::chrono::steady_clock::now();
                local.comp_s += std::chrono::duration<double>(t3 - t2).count();
                if (out_size != entry.raw_len) {
                    throw std::runtime_error("Decompressed size mismatch");
                }
                std::memcpy(raw_buf.data() + rank_offset_bytes,
                            chunk_raw.data() + chunk_offset,
                            copy_len);
            }

            chunk_raw_start = chunk_raw_end;
        }

        auto t4 = std::chrono::steady_clock::now();
        write_fully(out_fd,
                    raw_buf.data(),
                    raw_buf.size(),
                    static_cast<std::uint64_t>(rank_offset) * elem_size);
        auto t5 = std::chrono::steady_clock::now();
        local.io_write_s += std::chrono::duration<double>(t5 - t4).count();

        close(in_fd);
        close(out_fd);
    }

    Metrics agg = local;
    auto reduce_max = [&](double v) {
        double out = 0.0;
        MPI_Reduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        return out;
    };
    auto reduce_sum_u64 = [&](std::uint64_t v) {
        std::uint64_t out = 0;
        MPI_Reduce(&v, &out, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        return out;
    };
    auto reduce_max_u64 = [&](std::uint64_t v) {
        std::uint64_t out = 0;
        MPI_Reduce(&v, &out, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
        return out;
    };

    agg.io_read_s = reduce_max(local.io_read_s);
    agg.comp_s = reduce_max(local.comp_s);
    agg.io_write_s = reduce_max(local.io_write_s);
    agg.raw_bytes = reduce_sum_u64(local.raw_bytes);
    agg.comp_bytes = reduce_sum_u64(local.comp_bytes);
    const std::uint64_t max_raw_bytes = reduce_max_u64(local.raw_bytes);

    if (mpi.rank == 0) {
        const double raw_mb = static_cast<double>(agg.raw_bytes) / 1048576.0;
        const double comp_mb = static_cast<double>(agg.comp_bytes) / 1048576.0;
        const double local_raw_mb = static_cast<double>(max_raw_bytes) / 1048576.0;
        const double read_thr = agg.io_read_s > 0.0 ? comp_mb / agg.io_read_s : 0.0;
        const double decomp_thr = agg.comp_s > 0.0 ? raw_mb / agg.comp_s : 0.0;
        const double write_thr = agg.io_write_s > 0.0 ? raw_mb / agg.io_write_s : 0.0;
        const double decomp_write_s = agg.comp_s + agg.io_write_s;
        const double decomp_write_thr = decomp_write_s > 0.0 ? raw_mb / decomp_write_s : 0.0;
        const double read_thr_local = agg.io_read_s > 0.0 ? (comp_mb / mpi.ranks) / agg.io_read_s : 0.0;
        const double decomp_thr_local = agg.comp_s > 0.0 ? local_raw_mb / agg.comp_s : 0.0;
        const double write_thr_local = agg.io_write_s > 0.0 ? local_raw_mb / agg.io_write_s : 0.0;
        const double decomp_write_thr_local =
            decomp_write_s > 0.0 ? local_raw_mb / decomp_write_s : 0.0;
        const double ratio = agg.raw_bytes > 0 ? static_cast<double>(agg.comp_bytes) / agg.raw_bytes : 0.0;

        const std::string dtype_str = (params.dtype == mans::DataType::U16) ? "u2" : "u4";
        std::cout << kAnsiBold << "Mans mpi bench finished!" << kAnsiReset
                  << " Output: " << opts.output_bin << "\n";
        std::cout << kAnsiDim << "Config: " << kAnsiReset
                  << "mode=" << mode_to_string(opts.mode)
                  << ", dtype=" << dtype_str
                  << ", format=container"
                  << ", filter=" << (opts.filter == FilterId::Mans ? "mans" : "none")
                  << ", chunk_count=" << header.chunk_count
                  << ", ranks=" << mpi.ranks
                  << "\n";
        std::cout << kAnsiBold << "Decompress" << kAnsiReset << "\n";
        std::cout << "  " << kAnsiBlue << "IO read s" << kAnsiReset << ": "
                  << agg.io_read_s << " (" << read_thr << " MB/s)\n";
        std::cout << "  " << kAnsiBlue << "Decompress s" << kAnsiReset << ": "
                  << agg.comp_s << " (" << decomp_thr << " MB/s)\n";
        std::cout << "  " << kAnsiBlue << "IO write s" << kAnsiReset << ": "
                  << agg.io_write_s << " (" << write_thr << " MB/s)\n";
        std::cout << "  " << kAnsiBlue << "Decomp+IO write s" << kAnsiReset << ": "
                  << decomp_write_s << " (" << decomp_write_thr << " MB/s)\n";
        std::cout << "  Throughput (MB/s, decomp only): "
                  << kAnsiGreen << decomp_thr << kAnsiReset << "\n";
        std::cout << "  " << kAnsiYellow << "IO ratio" << kAnsiReset << ": "
                  << (agg.io_read_s + agg.io_write_s + agg.comp_s > 0.0
                          ? (agg.io_read_s + agg.io_write_s) /
                                (agg.io_read_s + agg.io_write_s + agg.comp_s)
                          : 0.0)
                  << "\n";
        std::cout << "  " << kAnsiYellow << "Ratio" << kAnsiReset << ": "
                  << ratio << "\n";
        std::cout << kAnsiDim << "  Local max (MB/s): read=" << read_thr_local
                  << ", decomp=" << decomp_thr_local
                  << ", write=" << write_thr_local
                  << ", decomp+write=" << decomp_write_thr_local
                  << kAnsiReset << "\n";

        const std::string csv_path = opts.metrics_csv.empty()
            ? (opts.input_bin + ".mpi_metrics.csv")
            : opts.metrics_csv;
        append_metrics_csv(csv_path, opts, total_elems, elem_size, mpi.ranks, agg);
        std::cout << "  metrics_csv:    " << csv_path << "\n";
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MpiContext mpi(argc, argv);

    RunOptions opts;
    try {
        const auto parsed = parse_args(argc, argv, mpi.rank);
        if (!parsed.has_value()) {
            return 1;
        }
        opts = *parsed;
    } catch (const std::exception& e) {
        if (mpi.rank == 0) {
            std::cerr << "Arg parse error: " << e.what() << "\n";
        }
        return 1;
    }

    if (opts.expected_ranks > 0 && opts.expected_ranks != mpi.ranks && mpi.rank == 0) {
        std::cerr << "[WARN] --ranks " << opts.expected_ranks
                  << " does not match MPI world size " << mpi.ranks << ". Using MPI size.\n";
    }

    if (opts.mode == BenchMode::Decompress) {
        return run_decompress_mode(mpi, opts);
    }

    std::optional<mans::h5::MansConfig> mans_cfg;
    mans::MansParams params{};
    params.backend = mans::Backend::CPU;
    params.dtype = mans::DataType::U32;
    params.adm_threshold = 4000U;
    params.adm_decide_threads = 16;

    if (opts.filter == FilterId::Mans) {
        mans_cfg.emplace();
        try {
            mans_cfg->load(opts.mans_config_file);
        } catch (const std::exception& e) {
            if (mpi.rank == 0) {
                std::cerr << "MANS config error: " << e.what() << "\n";
            }
            return 1;
        }
        params = mans_cfg->get_params();
        params.backend = mans::Backend::CPU;
    }

    if (opts.dtype_override.has_value()) {
        if (opts.filter == FilterId::Mans && params.dtype != *opts.dtype_override && mpi.rank == 0) {
            std::cerr << "[WARN] Overriding mans_config dtype with --dtype\n";
        }
        params.dtype = *opts.dtype_override;
    }

    std::uint32_t dtype = params.dtype;
    const std::size_t elem_size = (dtype == mans::DataType::U16) ? sizeof(std::uint16_t) : sizeof(std::uint32_t);

    mans::data_gen::SyntheticConfig synth_cfg{};
    mans::data_gen::GeneratorConfig gen_cfg{};

    if (opts.input_bin.empty()) {
        if (opts.dataset_config_file.empty()) {
            if (mpi.rank == 0) {
                std::cerr << "[Error] input_bin is empty; provide --dataset-config to generate data.\n";
            }
            return 1;
        }
        try {
            gen_cfg = mans::data_gen::load_generator_config(opts.dataset_config_file);
            synth_cfg = gen_cfg.synth;
        } catch (const std::exception& e) {
            if (mpi.rank == 0) {
                std::cerr << "Dataset config error: " << e.what() << "\n";
            }
            return 1;
        }
        if (opts.output_bin.empty() && !gen_cfg.output_bin.empty()) {
            opts.output_bin = gen_cfg.output_bin;
        }
        if (!opts.dtype_override.has_value() && opts.filter != FilterId::Mans) {
            dtype = gen_cfg.dtype;
        }
    }

    if (opts.output_bin.empty()) {
        if (mpi.rank == 0) {
            std::cerr << "[Error] output_bin is still empty after loading dataset_config_file.\n";
        }
        return 1;
    }

    if (dtype != mans::DataType::U16 && dtype != mans::DataType::U32) {
        if (mpi.rank == 0) {
            std::cerr << "[Error] Unsupported dtype value: " << dtype << "\n";
        }
        return 1;
    }

    const std::size_t elem_size_final = (dtype == mans::DataType::U16) ? sizeof(std::uint16_t) : sizeof(std::uint32_t);
    if (elem_size_final != elem_size) {
        params.dtype = dtype;
    }

    std::size_t total_elems = 0;
    try {
        if (!opts.input_bin.empty()) {
            const auto bytes = file_size_bytes(opts.input_bin);
            if (bytes % elem_size_final != 0) {
                throw std::runtime_error("Input file size not aligned to dtype");
            }
            total_elems = static_cast<std::size_t>(bytes / elem_size_final);
        } else {
            if (synth_cfg.size_mb <= 0.0) {
                synth_cfg.size_mb = 256.0;
            }
            total_elems = mans::data_gen::aligned_total_elements(
                synth_cfg.size_mb, elem_size_final, synth_cfg.block_size);
        }
    } catch (const std::exception& e) {
        if (mpi.rank == 0) {
            std::cerr << "Data sizing error: " << e.what() << "\n";
        }
        return 1;
    }

    std::size_t rank_offset = 0;
    std::size_t rank_count = 0;
    split_even(total_elems, mpi.rank, mpi.ranks, rank_offset, rank_count);

    if (mpi.rank == 0) {
        const std::string dtype_str = (dtype == mans::DataType::U16) ? "u2" : "u4";
        std::cout << "  " << kAnsiCyan << "OMP max threads" << kAnsiReset << ": "
                  << omp_get_max_threads() << "\n";
        std::cout << kAnsiBold << "Command-line arguments:" << kAnsiReset << "\n";
        std::cout << "  " << kAnsiCyan << "Mode" << kAnsiReset << ": " << mode_to_string(opts.mode) << "\n";
        std::cout << "  " << kAnsiCyan << "Input type" << kAnsiReset << ": " << dtype_str << "\n";
        std::cout << "  " << kAnsiCyan << "Input file" << kAnsiReset << ": "
                  << (opts.input_bin.empty() ? "(generated)" : opts.input_bin) << "\n";
        std::cout << "  " << kAnsiCyan << "Output file" << kAnsiReset << ": " << opts.output_bin << "\n";
        std::cout << "  " << kAnsiCyan << "Filter" << kAnsiReset << ": "
                  << (opts.filter == FilterId::Mans ? "mans" : "none") << "\n";
        if (opts.chunk_mb > 0.0) {
            std::cout << "  " << kAnsiCyan << "Chunk size (MB)" << kAnsiReset << ": " << opts.chunk_mb << "\n";
        } else {
            std::cout << "  " << kAnsiCyan << "Chunk size (MB)" << kAnsiReset << ": full rank\n";
        }
        std::cout << "  " << kAnsiCyan << "MPI ranks" << kAnsiReset << ": " << mpi.ranks << "\n";
        if (!opts.metrics_csv.empty()) {
            std::cout << "  " << kAnsiCyan << "CSV output" << kAnsiReset << ": " << opts.metrics_csv << "\n";
        }
        std::cout << "\n";
    }

    const double local_mb = static_cast<double>(rank_count) * elem_size_final / 1048576.0;
    std::vector<double> per_rank_mb;
    if (mpi.rank == 0) {
        per_rank_mb.resize(mpi.ranks, 0.0);
    }
    MPI_Gather(&local_mb, 1, MPI_DOUBLE,
               per_rank_mb.data(), 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    if (mpi.rank == 0) {
        std::cout << "[Rank Split]\n";
        std::cout << std::fixed << std::setprecision(3);
        for (int r = 0; r < mpi.ranks; ++r) {
            std::cout << "  rank " << r << ": " << per_rank_mb[r] << " MB\n";
        }
    }

    if (opts.raw_io && opts.filter != FilterId::None) {
        if (mpi.rank == 0) {
            std::cerr << "[Error] --raw-io requires --filter none\n";
        }
        return 1;
    }

    Metrics local{};
    local.raw_bytes = static_cast<std::uint64_t>(rank_count) * elem_size_final;

    auto reduce_max = [&](double v) {
        double out = 0.0;
        MPI_Reduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        return out;
    };
    auto reduce_sum_u64 = [&](std::uint64_t v) {
        std::uint64_t out = 0;
        MPI_Reduce(&v, &out, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        return out;
    };
    auto reduce_max_u64 = [&](std::uint64_t v) {
        std::uint64_t out = 0;
        MPI_Reduce(&v, &out, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
        return out;
    };

    std::vector<std::uint8_t> raw_bytes_buf;
    raw_bytes_buf.resize(rank_count * elem_size_final);

    if (rank_count > 0) {
        auto t0 = std::chrono::steady_clock::now();
        if (!opts.input_bin.empty()) {
            int fd = open(opts.input_bin.c_str(), O_RDONLY);
            if (fd < 0) {
                if (mpi.rank == 0) {
                    std::cerr << "[Error] Failed to open input_bin: " << opts.input_bin << "\n";
                }
                return 1;
            }
            read_fully(fd,
                       raw_bytes_buf.data(),
                       raw_bytes_buf.size(),
                       static_cast<std::uint64_t>(rank_offset) * elem_size_final);
            close(fd);
        } else {
            const std::uint32_t threshold = (opts.filter == FilterId::Mans && mans_cfg.has_value())
                ? mans_cfg->get_params().adm_threshold
                : gen_cfg.adm_threshold;
            if (dtype == mans::DataType::U16) {
                const auto slice = mans::data_gen::generate_synthetic_slice<std::uint16_t>(
                    threshold, synth_cfg, total_elems, rank_offset, rank_count);
                std::memcpy(raw_bytes_buf.data(), slice.data(), slice.size() * sizeof(std::uint16_t));
            } else {
                const auto slice = mans::data_gen::generate_synthetic_slice<std::uint32_t>(
                    threshold, synth_cfg, total_elems, rank_offset, rank_count);
                std::memcpy(raw_bytes_buf.data(), slice.data(), slice.size() * sizeof(std::uint32_t));
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        local.io_read_s = std::chrono::duration<double>(t1 - t0).count();
    }

    const bool use_container = !opts.raw_io;
    const double agg_io_read_s = reduce_max(local.io_read_s);
    std::uint64_t agg_raw_bytes = 0;
    MPI_Allreduce(&local.raw_bytes, &agg_raw_bytes, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    const std::uint64_t max_raw_bytes = reduce_max_u64(local.raw_bytes);

    std::size_t chunk_elements = rank_count;
    if (use_container) {
        if (opts.chunk_mb > 0.0) {
            const double bytes = opts.chunk_mb * 1048576.0;
            chunk_elements = static_cast<std::size_t>(bytes / elem_size_final);
            if (chunk_elements == 0) {
                chunk_elements = 1;
            }
            if (chunk_elements > rank_count) {
                chunk_elements = rank_count;
            }
        }
    }
    const auto chunks = use_container ? build_chunks(rank_count, chunk_elements) : std::vector<ChunkInfo>{};
    const std::uint64_t local_chunk_count = static_cast<std::uint64_t>(chunks.size());

    std::vector<std::uint64_t> chunk_counts;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    std::uint64_t total_chunks = 0;
    std::uint64_t prefix_chunks = 0;
    if (use_container) {
        if (mpi.rank == 0) {
            chunk_counts.resize(mpi.ranks, 0);
        }
        MPI_Gather(&local_chunk_count, 1, MPI_UINT64_T,
                   chunk_counts.data(), 1, MPI_UINT64_T,
                   0, MPI_COMM_WORLD);
        MPI_Exscan(&local_chunk_count, &prefix_chunks, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (mpi.rank == 0) {
            prefix_chunks = 0;
            for (const auto count : chunk_counts) {
                total_chunks += count;
            }
            recv_counts.resize(mpi.ranks, 0);
            recv_displs.resize(mpi.ranks, 0);
            std::uint64_t running = 0;
            for (int r = 0; r < mpi.ranks; ++r) {
                recv_counts[r] = static_cast<int>(chunk_counts[r]);
                recv_displs[r] = static_cast<int>(running);
                running += chunk_counts[r];
            }
        }
        MPI_Bcast(&total_chunks, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    }

    mans::container::ContainerHeader base_header{};
    if (use_container) {
        std::memcpy(base_header.magic, mans::container::kContainerMagic,
                    sizeof(mans::container::kContainerMagic));
        base_header.version = mans::container::kContainerVersion;
        base_header.dtype = (dtype == mans::DataType::U16) ? 1 : 2;
        base_header.reserved0 = 0;
        base_header.header_bytes = static_cast<std::uint16_t>(sizeof(mans::container::ContainerHeader));
        base_header.chunk_count = total_chunks;
        base_header.index_offset = sizeof(mans::container::ContainerHeader);
        base_header.data_offset = base_header.index_offset +
                                  total_chunks * sizeof(mans::container::IndexEntry);
        base_header.chunk_header_bytes = sizeof(mans::container::ChunkHeader);
        base_header.flags = (opts.filter == FilterId::None)
            ? mans::container::kFlagRawPayload
            : 0;
    }

    constexpr int kIters = 11;
    double total_comp_s = 0.0;
    double total_io_write_s = 0.0;
    std::uint64_t last_comp_bytes = 0;
    std::uint64_t local_comp_bytes_last = 0;

    MANS_TIMING_RESET();
    for (int iter = 0; iter < kIters; ++iter) {
        MANS_TIMING_RUN_SCOPE();
        MANS_TIMING_SCOPE("total");
        if (mpi.rank == 0) {
            const char* label = (opts.mode == BenchMode::Bench) ? "[Bench]" : "[Compress]";
            std::cout << "\r" << kAnsiDim << label << " iter "
                      << (iter + 1) << "/" << kIters << kAnsiReset << std::flush;
        }

        const std::size_t num_chunks = chunks.size();
        std::unique_ptr<std::uint64_t[]> local_comp_lens;
        std::unique_ptr<std::uint64_t[]> local_raw_lens;
        std::unique_ptr<PayloadBlock[]> payloads;
        std::uint64_t local_payload_bytes = 0;
        std::uint64_t local_block_bytes = 0;
        double local_comp_s = 0.0;
        double local_io_write_s = 0.0;

        if (rank_count > 0 && use_container) {
            if (num_chunks > 0) {
                local_comp_lens = std::make_unique<std::uint64_t[]>(num_chunks);
                local_raw_lens = std::make_unique<std::uint64_t[]>(num_chunks);
                if (opts.filter == FilterId::Mans) {
                    payloads = std::make_unique<PayloadBlock[]>(num_chunks);
                }
            }

            auto t2 = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < num_chunks; ++i) {
                const auto& chunk = chunks[i];
                const std::size_t raw_bytes = chunk.len * elem_size_final;
                if (opts.filter == FilterId::None) {
                    local_comp_lens[i] = static_cast<std::uint64_t>(raw_bytes);
                    local_raw_lens[i] = static_cast<std::uint64_t>(raw_bytes);
                    local_payload_bytes += raw_bytes;
                    local_block_bytes += sizeof(mans::container::ChunkHeader) + raw_bytes;
                    continue;
                }

                const std::size_t max_out_size =
                    mans::get_mans_max_compress_bytes_p(chunk.len, params);
                auto* payload_buf = static_cast<std::uint8_t*>(std::malloc(max_out_size));
                if (!payload_buf) {
                    throw std::runtime_error("Failed to allocate payload buffer");
                }
                std::size_t out_size = max_out_size;
                if (dtype == mans::DataType::U16) {
                    const auto* ptr = reinterpret_cast<const std::uint16_t*>(raw_bytes_buf.data());
                    mans::cpu::compress_internal(ptr + chunk.offset,
                                                 chunk.len,
                                                 params,
                                                 payload_buf,
                                                 out_size,
                                                 false,
                                                 "");
                } else {
                    const auto* ptr = reinterpret_cast<const std::uint32_t*>(raw_bytes_buf.data());
                    mans::cpu::compress_internal(ptr + chunk.offset,
                                                 chunk.len,
                                                 params,
                                                 payload_buf,
                                                 out_size,
                                                 false,
                                                 "");
                }
                if (out_size == 0) {
                    std::free(payload_buf);
                    throw std::runtime_error("Compression failed for chunk");
                }
                payloads[i].data.reset(payload_buf);
                payloads[i].size = out_size;
                local_comp_lens[i] = static_cast<std::uint64_t>(out_size);
                local_raw_lens[i] = static_cast<std::uint64_t>(raw_bytes);
                local_payload_bytes += out_size;
                local_block_bytes += sizeof(mans::container::ChunkHeader) + out_size;
            }
            auto t3 = std::chrono::steady_clock::now();
            local_comp_s = std::chrono::duration<double>(t3 - t2).count();
        }

        std::uint64_t local_comp_bytes = use_container ? local_payload_bytes : local.raw_bytes;
        local_comp_bytes_last = local_comp_bytes;

        std::uint64_t prefix_block_bytes = 0;
        if (use_container) {
            MPI_Exscan(&local_block_bytes, &prefix_block_bytes, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
            if (mpi.rank == 0) {
                prefix_block_bytes = 0;
            }
        }

        std::unique_ptr<std::uint64_t[]> all_comp_lens;
        std::unique_ptr<std::uint64_t[]> all_raw_lens;
        std::uint64_t total_block_bytes = 0;
        if (use_container) {
            total_block_bytes = reduce_sum_u64(local_block_bytes);
            if (mpi.rank == 0) {
                const std::size_t total_chunks_size = static_cast<std::size_t>(total_chunks);
                if (total_chunks_size > 0) {
                    all_comp_lens = std::make_unique<std::uint64_t[]>(total_chunks_size);
                    all_raw_lens = std::make_unique<std::uint64_t[]>(total_chunks_size);
                }
            }
            MPI_Gatherv(local_comp_lens.get(),
                        static_cast<int>(num_chunks),
                        MPI_UINT64_T,
                        all_comp_lens.get(),
                        recv_counts.data(),
                        recv_displs.data(),
                        MPI_UINT64_T,
                        0,
                        MPI_COMM_WORLD);
            MPI_Gatherv(local_raw_lens.get(),
                        static_cast<int>(num_chunks),
                        MPI_UINT64_T,
                        all_raw_lens.get(),
                        recv_counts.data(),
                        recv_displs.data(),
                        MPI_UINT64_T,
                        0,
                        MPI_COMM_WORLD);
            if (mpi.rank == 0) {
                std::unique_ptr<mans::container::IndexEntry[]> index;
                const std::size_t total_chunks_size = static_cast<std::size_t>(total_chunks);
                if (total_chunks_size > 0) {
                    index = std::make_unique<mans::container::IndexEntry[]>(total_chunks_size);
                }
                std::uint64_t offset = base_header.data_offset;
                for (std::uint64_t i = 0; i < total_chunks; ++i) {
                    mans::container::IndexEntry entry{};
                    entry.offset = offset;
                    entry.comp_len = all_comp_lens[static_cast<std::size_t>(i)];
                    entry.raw_len = all_raw_lens[static_cast<std::size_t>(i)];
                    index[static_cast<std::size_t>(i)] = entry;
                    offset += sizeof(mans::container::ChunkHeader) + entry.comp_len;
                }
                const std::uint64_t expected_total = offset - base_header.data_offset;
                if (expected_total != total_block_bytes) {
                    std::cerr << "[WARN] Total payload bytes mismatch: expected "
                              << expected_total << " vs sum " << total_block_bytes << "\n";
                }
                const std::uint64_t file_bytes =
                    static_cast<std::uint64_t>(base_header.data_offset) + expected_total;

                int fd = open(opts.output_bin.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
                if (fd < 0) {
                    std::cerr << "[Error] Failed to open output_bin: " << opts.output_bin << "\n";
                    return 1;
                }
                write_fully(fd, &base_header, sizeof(base_header), 0);
                write_fully(fd,
                            index.get(),
                            total_chunks_size * sizeof(mans::container::IndexEntry),
                            static_cast<std::uint64_t>(base_header.index_offset));
                if (ftruncate(fd, static_cast<off_t>(file_bytes)) != 0) {
                    std::cerr << "[WARN] Failed to ftruncate output file.\n";
                }
                close(fd);
            }
        } else {
            if (mpi.rank == 0) {
                int fd = open(opts.output_bin.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
                if (fd < 0) {
                    std::cerr << "[Error] Failed to open output_bin: " << opts.output_bin << "\n";
                    return 1;
                }
                if (ftruncate(fd, static_cast<off_t>(agg_raw_bytes)) != 0) {
                    std::cerr << "[WARN] Failed to ftruncate output file.\n";
                }
                close(fd);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank_count > 0) {
            int fd = open(opts.output_bin.c_str(), O_WRONLY);
            if (fd < 0) {
                if (mpi.rank == 0) {
                    std::cerr << "[Error] Failed to open output_bin for writing: " << opts.output_bin << "\n";
                }
                return 1;
            }
            auto t0 = std::chrono::steady_clock::now();
            if (use_container) {
                std::uint64_t write_offset = base_header.data_offset + prefix_block_bytes;
                std::uint64_t chunk_index = prefix_chunks;
                for (std::size_t i = 0; i < num_chunks; ++i) {
                    mans::container::ChunkHeader chunk_header{};
                    std::memcpy(chunk_header.magic, mans::container::kChunkMagic,
                                sizeof(mans::container::kChunkMagic));
                    chunk_header.version = mans::container::kChunkVersion;
                    chunk_header.header_bytes = static_cast<std::uint16_t>(sizeof(mans::container::ChunkHeader));
                    chunk_header.comp_len = local_comp_lens[i];
                    chunk_header.raw_len = local_raw_lens[i];
                    chunk_header.chunk_index = chunk_index++;

                    write_fully(fd, &chunk_header, sizeof(chunk_header), write_offset);
                    write_offset += sizeof(chunk_header);
                    if (opts.filter == FilterId::None) {
                        const std::size_t raw_offset = chunks[i].offset * elem_size_final;
                        const std::size_t raw_size = chunks[i].len * elem_size_final;
                        write_fully(fd,
                                    raw_bytes_buf.data() + raw_offset,
                                    raw_size,
                                    write_offset);
                        write_offset += raw_size;
                    } else {
                        write_fully(fd, payloads[i].data.get(), payloads[i].size, write_offset);
                        write_offset += payloads[i].size;
                    }
                }
            } else {
                write_fully(fd,
                            raw_bytes_buf.data(),
                            raw_bytes_buf.size(),
                            static_cast<std::uint64_t>(rank_offset) * elem_size_final);
            }
            auto t1 = std::chrono::steady_clock::now();
            local_io_write_s = std::chrono::duration<double>(t1 - t0).count();
            close(fd);
        }

        const double iter_comp_s = reduce_max(local_comp_s);
        const double iter_io_write_s = reduce_max(local_io_write_s);
        const std::uint64_t iter_comp_bytes = reduce_sum_u64(local_comp_bytes);

        if (mpi.rank == 0) {
            if (iter > 0) {
                total_comp_s += iter_comp_s;
                total_io_write_s += iter_io_write_s;
            }
            if (iter == kIters - 1) {
                last_comp_bytes = iter_comp_bytes;
            }
        }
    }

    if (mpi.rank == 0) {
        std::cout << "\n";
    }

    const double denom = static_cast<double>(kIters - 1);
    const double avg_comp_s = total_comp_s / denom;
    const double avg_io_write_s = total_io_write_s / denom;
    const double comp_write_s = avg_comp_s + avg_io_write_s;
    const std::string timing_comp_path = opts.output_bin + ".comp.csv";
    if (mpi.rank == 0) {
        MANS_TIMING_DUMP(timing_comp_path);
    }

    if (mpi.rank == 0) {
        const double raw_mb = static_cast<double>(agg_raw_bytes) / 1048576.0;
        const double local_raw_mb = static_cast<double>(max_raw_bytes) / 1048576.0;
        const double read_thr = agg_io_read_s > 0.0 ? raw_mb / agg_io_read_s : 0.0;
        const double comp_thr = avg_comp_s > 0.0 ? raw_mb / avg_comp_s : 0.0;
        const double write_thr = avg_io_write_s > 0.0 ? raw_mb / avg_io_write_s : 0.0;
        const double comp_write_thr = comp_write_s > 0.0 ? raw_mb / comp_write_s : 0.0;
        const double read_thr_local = agg_io_read_s > 0.0 ? local_raw_mb / agg_io_read_s : 0.0;
        const double comp_thr_local = avg_comp_s > 0.0 ? local_raw_mb / avg_comp_s : 0.0;
        const double write_thr_local = avg_io_write_s > 0.0 ? local_raw_mb / avg_io_write_s : 0.0;
        const double comp_write_thr_local = comp_write_s > 0.0 ? local_raw_mb / comp_write_s : 0.0;
        const double ratio = agg_raw_bytes > 0 ? static_cast<double>(last_comp_bytes) / agg_raw_bytes : 0.0;

        const std::string dtype_str = (dtype == mans::DataType::U16) ? "u2" : "u4";
        std::cout << kAnsiBold << "Mans mpi bench finished!" << kAnsiReset
                  << " Output: " << opts.output_bin << "\n";
        std::cout << kAnsiDim << "Config: " << kAnsiReset
                  << "mode=" << mode_to_string(opts.mode)
                  << ", dtype=" << dtype_str
                  << ", format=" << (use_container ? "container" : "raw")
                  << ", filter=" << (opts.filter == FilterId::Mans ? "mans" : "none")
                  << ", chunk_mb=" << opts.chunk_mb
                  << ", chunk_count=" << (use_container ? total_chunks : 0)
                  << ", ranks=" << mpi.ranks
                  << "\n";
        std::cout << kAnsiBold << "Compress" << kAnsiReset << "\n";
        std::cout << "  " << kAnsiBlue << "IO read s" << kAnsiReset << ": "
                  << agg_io_read_s << " (" << read_thr << " MB/s)\n";
        std::cout << "  " << kAnsiBlue << "Compress s" << kAnsiReset << ": "
                  << avg_comp_s << " (" << comp_thr << " MB/s)\n";
        std::cout << "  " << kAnsiBlue << "IO write s" << kAnsiReset << ": "
                  << avg_io_write_s << " (" << write_thr << " MB/s)\n";
        std::cout << "  " << kAnsiBlue << "Comp+IO write s" << kAnsiReset << ": "
                  << comp_write_s << " (" << comp_write_thr << " MB/s)\n";
        std::cout << "  Throughput (MB/s, comp only): "
                  << kAnsiGreen << comp_thr << kAnsiReset << "\n";
        std::cout << "  " << kAnsiYellow << "IO ratio" << kAnsiReset << ": "
                  << (agg_io_read_s + avg_io_write_s + avg_comp_s > 0.0
                          ? (agg_io_read_s + avg_io_write_s) /
                                (agg_io_read_s + avg_io_write_s + avg_comp_s)
                          : 0.0)
                  << "\n";
        std::cout << "  " << kAnsiYellow << "Ratio" << kAnsiReset << ": "
                  << ratio << "\n";
        std::cout << kAnsiDim << "  Local max (MB/s): read=" << read_thr_local
                  << ", comp=" << comp_thr_local
                  << ", write=" << write_thr_local
                  << ", comp+write=" << comp_write_thr_local
                  << kAnsiReset << "\n";

        const std::string csv_path = opts.metrics_csv.empty()
            ? (opts.output_bin + ".mpi_metrics.csv")
            : opts.metrics_csv;
        Metrics agg{};
        agg.io_read_s = agg_io_read_s;
        agg.comp_s = avg_comp_s;
        agg.io_write_s = avg_io_write_s;
        agg.raw_bytes = agg_raw_bytes;
        agg.comp_bytes = last_comp_bytes;
        append_metrics_csv(csv_path, opts, total_elems, elem_size_final, mpi.ranks, agg);
        std::cout << "  metrics_csv:    " << csv_path << "\n";
        std::cout << "  timing_csv:     " << timing_comp_path << "\n";
    }

    return 0;
}
