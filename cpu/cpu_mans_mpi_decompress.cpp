// compiler: mpicxx -std=c++17 -O3 cpu_mans_mpi_decompress.cpp -o cpu_mans_mpi_decompress
// exec    : mpirun -n 4 ./cpu_mans_mpi_decompress --config /path/to/mpi_mans.cfg

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
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

#include "../mans_api.hpp"
#include "../mans_defs.h"
#include "../mans_timing.h"
#include "adm/adm_utils.h"
#include "../tools/H5Z-MANS/include/H5Z-MANS_config.h"

namespace {

constexpr char kMagic[8] = {'M','A','N','S','M','P','I','1'};
constexpr std::uint32_t kVersion = 2;

enum class FilterId : std::uint32_t {
    None = 0,
    Mans = 1,
};

struct RunOptions {
    std::string input_bin;
    std::string output_bin;
    std::string mans_config_file;
    std::string metrics_csv;

    FilterId filter = FilterId::Mans;
    bool filter_seen = false;
    int expected_ranks = -1;
    bool mode_seen = false;
};

struct Metrics {
    double io_read_s = 0.0;
    double comp_s = 0.0;
    double io_write_s = 0.0;
    std::uint64_t raw_bytes = 0;
    std::uint64_t comp_bytes = 0;
};

struct MansMpiHeader {
    char magic[8];
    std::uint32_t version = 0;
    std::uint32_t filter = 0;
    std::uint32_t dtype = 0;
    std::uint32_t elem_size = 0;
    std::uint64_t total_elems = 0;
    std::uint32_t ranks = 0;
    std::uint32_t reserved = 0;
    std::uint64_t total_comp_bytes = 0;
};

static_assert(sizeof(MansMpiHeader) == 48, "Unexpected header size");

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
            if (key == "input" || key == "input_bin" || key == "output_h5") {
                opts.input_bin = val;
                continue;
            }
            if (key == "output" || key == "output_bin") {
                opts.output_bin = val;
                continue;
            }
            if (key == "filter") {
                opts.filter = parse_filter(val);
                opts.filter_seen = true;
                continue;
            }
            if (key == "mode") {
                opts.mode_seen = true;
                if (val != "decompress") {
                    throw std::runtime_error("mode must be 'decompress' for cpu_mans_mpi_decompress");
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
            if (key == "csv" || key == "metrics_csv") {
                opts.metrics_csv = val;
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
        if (arg == "--filter") {
            opts.filter = parse_filter(need_value("--filter"));
            opts.filter_seen = true;
            continue;
        }
        if (arg == "--mode") {
            opts.mode_seen = true;
            const auto mode_val = need_value("--mode");
            if (mode_val != "decompress") {
                throw std::runtime_error("mode must be 'decompress' for cpu_mans_mpi_decompress");
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
        if (arg == "--csv") {
            opts.metrics_csv = need_value("--csv");
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            if (rank == 0) {
                std::cerr << "Usage:\n  " << argv[0]
                          << " --config mpi_mans.cfg [--input in.bin] [--output out.bin]\\\n"
                             "         [--filter mans|none] [--mans-config mans.cfg] [--csv metrics.csv]\n";
            }
            return std::nullopt;
        }

        throw std::runtime_error("Unknown argument: " + arg);
    }

    if (opts.input_bin.empty() || opts.output_bin.empty()) {
        if (rank == 0) {
            std::cerr << "[Error] input_bin and output_bin are required.\n";
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
               "io_read_s,comp_s,io_write_s,read_thr_mb_s,comp_thr_mb_s,write_thr_mb_s,ratio\n";
    }

    const double raw_mb = static_cast<double>(agg.raw_bytes) / 1048576.0;
    const double comp_mb = static_cast<double>(agg.comp_bytes) / 1048576.0;
    const double read_thr = agg.io_read_s > 0.0 ? comp_mb / agg.io_read_s : 0.0;
    const double comp_thr = agg.comp_s > 0.0 ? raw_mb / agg.comp_s : 0.0;
    const double write_thr = agg.io_write_s > 0.0 ? raw_mb / agg.io_write_s : 0.0;
    const double ratio = agg.raw_bytes > 0 ? static_cast<double>(agg.comp_bytes) / agg.raw_bytes : 0.0;

    out << "decompress,"
        << (opts.filter == FilterId::Mans ? "mans" : "none") << ","
        << 0.0 << ","
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
        << ratio
        << "\n";
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

    MansMpiHeader header{};
    std::vector<std::uint64_t> all_chunk_sizes;
    if (mpi.rank == 0) {
        int fd = open(opts.input_bin.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "[Error] Failed to open input_bin: " << opts.input_bin << "\n";
            return 1;
        }
        read_fully(fd, &header, sizeof(header), 0);
        if (std::memcmp(header.magic, kMagic, sizeof(header.magic)) != 0 ||
            header.version != kVersion) {
            std::cerr << "[Error] Invalid input header (magic/version mismatch)\n";
            close(fd);
            return 1;
        }
        all_chunk_sizes.resize(header.ranks);
        read_fully(fd,
                   all_chunk_sizes.data(),
                   static_cast<std::size_t>(header.ranks) * sizeof(std::uint64_t),
                   sizeof(header));
        close(fd);
    }

    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (mpi.rank != 0) {
        all_chunk_sizes.resize(header.ranks);
    }
    MPI_Bcast(all_chunk_sizes.data(),
              static_cast<int>(header.ranks),
              MPI_UINT64_T,
              0,
              MPI_COMM_WORLD);

    if (header.ranks != static_cast<std::uint32_t>(mpi.ranks)) {
        if (mpi.rank == 0) {
            std::cerr << "[Error] MPI ranks (" << mpi.ranks
                      << ") do not match header ranks (" << header.ranks << ").\n";
        }
        return 1;
    }
    if (opts.expected_ranks > 0 && opts.expected_ranks != mpi.ranks && mpi.rank == 0) {
        std::cerr << "[WARN] --ranks " << opts.expected_ranks
                  << " does not match MPI world size " << mpi.ranks << ".\n";
    }

    const auto header_filter = static_cast<FilterId>(header.filter);
    if (!opts.filter_seen) {
        opts.filter = header_filter;
    } else if (opts.filter != header_filter) {
        if (mpi.rank == 0) {
            std::cerr << "[Error] Filter mismatch: config says "
                      << (opts.filter == FilterId::Mans ? "mans" : "none")
                      << ", header says " << (header_filter == FilterId::Mans ? "mans" : "none")
                      << "\n";
        }
        return 1;
    }

    if (opts.filter == FilterId::Mans && opts.mans_config_file.empty()) {
        if (mpi.rank == 0) {
            std::cerr << "[Error] --mans-config is required when header filter is mans\n";
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

    params.dtype = header.dtype;
    const std::size_t elem_size = header.elem_size;
    const std::size_t total_elems = static_cast<std::size_t>(header.total_elems);

    std::size_t rank_offset = 0;
    std::size_t rank_count = 0;
    split_even(total_elems, mpi.rank, mpi.ranks, rank_offset, rank_count);

    const std::uint64_t local_comp_bytes = all_chunk_sizes[mpi.rank];

    std::vector<std::uint8_t> mans_intermediate;
    if (opts.filter == FilterId::Mans && rank_count > 0) {
        std::size_t adm_cap = 0;
        if (params.dtype == mans::DataType::U16) {
            adm_cap = adm_max_compressed_size<std::uint16_t>(rank_count);
        } else {
            adm_cap = adm_max_compressed_size<std::uint32_t>(rank_count);
        }
        const std::size_t raw_bytes = rank_count * elem_size;
        const std::size_t cap = std::max(adm_cap, raw_bytes);
        mans_intermediate.resize(cap);
    }

    std::uint64_t prefix_bytes = 0;
    MPI_Exscan(&local_comp_bytes, &prefix_bytes, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (mpi.rank == 0) {
        prefix_bytes = 0;
    }

    std::uint64_t total_comp_bytes = 0;
    MPI_Reduce(&local_comp_bytes, &total_comp_bytes, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    const std::uint64_t header_bytes = sizeof(MansMpiHeader) + static_cast<std::uint64_t>(header.ranks) * sizeof(std::uint64_t);

    if (mpi.rank == 0) {
        int fd = open(opts.output_bin.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (fd < 0) {
            std::cerr << "[Error] Failed to create output_bin: " << opts.output_bin << "\n";
            return 1;
        }
        const std::uint64_t total_raw_bytes = static_cast<std::uint64_t>(total_elems) * elem_size;
        if (ftruncate(fd, static_cast<off_t>(total_raw_bytes)) != 0) {
            std::cerr << "[WARN] Failed to ftruncate output file.\n";
        }
        close(fd);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    Metrics local{};
    local.raw_bytes = static_cast<std::uint64_t>(rank_count) * elem_size;
    local.comp_bytes = local_comp_bytes;

    if (rank_count > 0 && local_comp_bytes > 0) {
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

        std::vector<std::uint8_t> comp_buf;
        comp_buf.resize(local_comp_bytes);

        auto t0 = std::chrono::steady_clock::now();
        read_fully(in_fd, comp_buf.data(), comp_buf.size(), header_bytes + prefix_bytes);
        auto t1 = std::chrono::steady_clock::now();
        local.io_read_s += std::chrono::duration<double>(t1 - t0).count();

        const std::size_t raw_bytes = rank_count * elem_size;
        std::vector<std::uint8_t> raw_buf;
        raw_buf.resize(raw_bytes);

        auto t2 = std::chrono::steady_clock::now();
        if (opts.filter == FilterId::None) {
            if (comp_buf.size() != raw_bytes) {
                throw std::runtime_error("Compressed size mismatch for filter=none");
            }
            std::memcpy(raw_buf.data(), comp_buf.data(), raw_bytes);
        } else {
            std::size_t out_size = raw_bytes;
            auto* intermediate_ptr = mans_intermediate.empty() ? nullptr : mans_intermediate.data();
            const std::size_t intermediate_cap = mans_intermediate.size();
            mans::decompress(comp_buf.data(),
                             comp_buf.size(),
                             params,
                             raw_buf.data(),
                             out_size,
                             intermediate_ptr,
                             intermediate_cap);
            if (out_size != raw_bytes) {
                raw_buf.resize(out_size);
            }
        }
        auto t3 = std::chrono::steady_clock::now();
        local.comp_s += std::chrono::duration<double>(t3 - t2).count();

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

    agg.io_read_s = reduce_max(local.io_read_s);
    agg.comp_s = reduce_max(local.comp_s);
    agg.io_write_s = reduce_max(local.io_write_s);
    agg.raw_bytes = reduce_sum_u64(local.raw_bytes);
    agg.comp_bytes = reduce_sum_u64(local.comp_bytes);

    if (mpi.rank == 0) {
        const double raw_mb = static_cast<double>(agg.raw_bytes) / 1048576.0;
        const double comp_mb = static_cast<double>(agg.comp_bytes) / 1048576.0;
        const double read_thr = agg.io_read_s > 0.0 ? comp_mb / agg.io_read_s : 0.0;
        const double comp_thr = agg.comp_s > 0.0 ? raw_mb / agg.comp_s : 0.0;
        const double write_thr = agg.io_write_s > 0.0 ? raw_mb / agg.io_write_s : 0.0;
        const double ratio = agg.raw_bytes > 0 ? static_cast<double>(agg.comp_bytes) / agg.raw_bytes : 0.0;

        std::cout << "\n[Summary]\n";
        std::cout << "  comp_read_s:    " << agg.io_read_s << " (" << read_thr << " MB/s)\n";
        std::cout << "  decompress_s:   " << agg.comp_s << " (" << comp_thr << " MB/s)\n";
        std::cout << "  raw_write_s:    " << agg.io_write_s << " (" << write_thr << " MB/s)\n";
        std::cout << "  ratio:          " << ratio << " (comp/raw)\n";

        const std::string csv_path = opts.metrics_csv.empty()
            ? (opts.input_bin + ".mpi_metrics.csv")
            : opts.metrics_csv;
        append_metrics_csv(csv_path, opts, total_elems, elem_size, mpi.ranks, agg);
        std::cout << "  metrics_csv:    " << csv_path << "\n";
    }

    (void)total_comp_bytes;
    return 0;
}
