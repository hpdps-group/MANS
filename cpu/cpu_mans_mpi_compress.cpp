// compiler: mpicxx -std=c++17 -O3 cpu_mans_mpi_compress.cpp -o cpu_mans_mpi_compress
// exec    : mpirun -n 4 ./cpu_mans_mpi_compress --config /path/to/mpi_mans.cfg

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
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
#include "../mans_data_gen.h"
#include "../mans_defs.h"
#include "../mans_timing.h"
#include "adm/adm_utils.h"
#include "mans_cpu.h"
#include "../tools/H5Z-MANS/include/H5Z-MANS_config.h"

namespace {

constexpr double kDefaultChunkMb = 32.0;
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
    std::string dataset_config_file;
    std::string metrics_csv;

    double chunk_mb = kDefaultChunkMb;
    FilterId filter = FilterId::Mans;
    int expected_ranks = -1;
    std::optional<std::uint32_t> dtype_override;
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

struct ChunkInfo {
    std::size_t offset = 0;
    std::size_t len = 0;
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
                continue;
            }
            if (key == "mode") {
                opts.mode_seen = true;
                if (val != "compress") {
                    throw std::runtime_error("mode must be 'compress' for cpu_mans_mpi_compress");
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
            continue;
        }
        if (arg == "--mode") {
            opts.mode_seen = true;
            const auto mode_val = need_value("--mode");
            if (mode_val != "compress") {
                throw std::runtime_error("mode must be 'compress' for cpu_mans_mpi_compress");
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
        if (arg == "--help" || arg == "-h") {
            if (rank == 0) {
                std::cerr << "Usage:\n  " << argv[0]
                          << " --config mpi_mans.cfg [--input raw.bin] [--output out.bin]\\\n"
                             "         [--chunk-mb MB] [--filter mans|none] [--mans-config mans.cfg]\\\n"
                             "         [--dataset-config synth.cfg] [--dtype u16|u32] [--csv metrics.csv]\n";
            }
            return std::nullopt;
        }

        throw std::runtime_error("Unknown argument: " + arg);
    }

    if (opts.output_bin.empty() && opts.dataset_config_file.empty()) {
        if (rank == 0) {
            std::cerr << "[Error] output_bin is required (or provide dataset_config_file with output_bin).\n";
        }
        return std::nullopt;
    }

    if (opts.filter == FilterId::Mans && opts.mans_config_file.empty()) {
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
    const double read_thr = agg.io_read_s > 0.0 ? raw_mb / agg.io_read_s : 0.0;
    const double comp_thr = agg.comp_s > 0.0 ? raw_mb / agg.comp_s : 0.0;
    const double write_thr = agg.io_write_s > 0.0 ? raw_mb / agg.io_write_s : 0.0;
    const double comp_write_s = agg.comp_s + agg.io_write_s;
    const double comp_write_thr = comp_write_s > 0.0 ? raw_mb / comp_write_s : 0.0;
    const double ratio = agg.raw_bytes > 0 ? static_cast<double>(agg.comp_bytes) / agg.raw_bytes : 0.0;

    out << "compress,"
        << (opts.filter == FilterId::Mans ? "mans" : "none") << ","
        << opts.chunk_mb << ","
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

    Metrics local{};
    local.raw_bytes = static_cast<std::uint64_t>(rank_count) * elem_size_final;

    std::vector<std::uint8_t> comp_data;

    if (rank_count > 0) {
        std::vector<std::uint8_t> raw_bytes_buf;
        raw_bytes_buf.resize(rank_count * elem_size_final);

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

        double comp_s = 0.0;
        if (opts.filter == FilterId::None) {
            comp_data.resize(raw_bytes_buf.size());
            auto t2 = std::chrono::steady_clock::now();
            std::memcpy(comp_data.data(), raw_bytes_buf.data(), raw_bytes_buf.size());
            auto t3 = std::chrono::steady_clock::now();
            comp_s = std::chrono::duration<double>(t3 - t2).count();
        } else {
            std::size_t chunk_elements = rank_count;
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
            const auto chunks = build_chunks(rank_count, chunk_elements);

            std::size_t max_chunk_len = 0;
            std::size_t out_cap = 0;
            for (const auto& chunk : chunks) {
                out_cap += mans::get_mans_max_compress_bytes_p(chunk.len, params);
                if (chunk.len > max_chunk_len) {
                    max_chunk_len = chunk.len;
                }
            }
            comp_data.resize(out_cap);

            std::vector<std::uint8_t> mans_intermediate;
            if (max_chunk_len > 0) {
                std::size_t adm_cap = 0;
                if (dtype == mans::DataType::U16) {
                    adm_cap = adm_max_compressed_size<std::uint16_t>(max_chunk_len);
                } else {
                    adm_cap = adm_max_compressed_size<std::uint32_t>(max_chunk_len);
                }
                mans_intermediate.resize(adm_cap);
            }

            auto* intermediate_ptr = mans_intermediate.empty() ? nullptr : mans_intermediate.data();
            const std::size_t intermediate_cap = mans_intermediate.size();

            std::size_t offset = 0;
            auto t2 = std::chrono::steady_clock::now();
            for (const auto& chunk : chunks) {
                std::size_t out_size = out_cap - offset;
                if (dtype == mans::DataType::U16) {
                    const auto* ptr = reinterpret_cast<const std::uint16_t*>(raw_bytes_buf.data());
                    mans::cpu::compress_internal(ptr + chunk.offset,
                                                 chunk.len,
                                                 params,
                                                 comp_data.data() + offset,
                                                 out_size,
                                                 false,
                                                 "",
                                                 intermediate_ptr,
                                                 intermediate_cap
                                                );
                } else {
                    const auto* ptr = reinterpret_cast<const std::uint32_t*>(raw_bytes_buf.data());
                    mans::cpu::compress_internal(ptr + chunk.offset,
                                                 chunk.len,
                                                 params,
                                                 comp_data.data() + offset,
                                                 out_size,
                                                 false,
                                                 "",
                                                 intermediate_ptr,
                                                 intermediate_cap);
                }
                offset += out_size;
            }
            auto t3 = std::chrono::steady_clock::now();
            comp_s = std::chrono::duration<double>(t3 - t2).count();
            comp_data.resize(offset);
        }
        local.comp_s = comp_s;
    }

    local.comp_bytes = static_cast<std::uint64_t>(comp_data.size());

    std::uint64_t local_comp_bytes = static_cast<std::uint64_t>(comp_data.size());
    std::vector<std::uint64_t> comp_sizes;
    if (mpi.rank == 0) {
        comp_sizes.resize(mpi.ranks, 0);
    }
    MPI_Gather(&local_comp_bytes, 1, MPI_UINT64_T,
               comp_sizes.data(), 1, MPI_UINT64_T,
               0, MPI_COMM_WORLD);

    std::uint64_t prefix_bytes = 0;
    MPI_Exscan(&local_comp_bytes, &prefix_bytes, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (mpi.rank == 0) {
        prefix_bytes = 0;
    }

    std::uint64_t total_comp_bytes = 0;
    MPI_Reduce(&local_comp_bytes, &total_comp_bytes, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    std::uint64_t header_bytes = sizeof(MansMpiHeader) + static_cast<std::uint64_t>(mpi.ranks) * sizeof(std::uint64_t);

    if (mpi.rank == 0) {
        MansMpiHeader header{};
        std::memcpy(header.magic, kMagic, sizeof(header.magic));
        header.version = kVersion;
        header.filter = static_cast<std::uint32_t>(opts.filter);
        header.dtype = dtype;
        header.elem_size = static_cast<std::uint32_t>(elem_size_final);
        header.total_elems = total_elems;
        header.ranks = static_cast<std::uint32_t>(mpi.ranks);
        header.total_comp_bytes = total_comp_bytes;

        int fd = open(opts.output_bin.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (fd < 0) {
            std::cerr << "[Error] Failed to open output_bin: " << opts.output_bin << "\n";
            return 1;
        }
        write_fully(fd, &header, sizeof(header), 0);
        write_fully(fd,
                    comp_sizes.data(),
                    static_cast<std::size_t>(mpi.ranks) * sizeof(std::uint64_t),
                    sizeof(header));
        if (ftruncate(fd, static_cast<off_t>(header_bytes + total_comp_bytes)) != 0) {
            std::cerr << "[WARN] Failed to ftruncate output file.\n";
        }
        close(fd);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (!comp_data.empty()) {
        int fd = open(opts.output_bin.c_str(), O_WRONLY);
        if (fd < 0) {
            if (mpi.rank == 0) {
                std::cerr << "[Error] Failed to open output_bin for writing: " << opts.output_bin << "\n";
            }
            return 1;
        }
        auto t0 = std::chrono::steady_clock::now();
        write_fully(fd, comp_data.data(), comp_data.size(), header_bytes + prefix_bytes);
        auto t1 = std::chrono::steady_clock::now();
        local.io_write_s = std::chrono::duration<double>(t1 - t0).count();
        close(fd);
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
    const std::uint64_t max_comp_bytes = reduce_max_u64(local.comp_bytes);

    if (mpi.rank == 0) {
        const double raw_mb = static_cast<double>(agg.raw_bytes) / 1048576.0;
        const double comp_mb = static_cast<double>(agg.comp_bytes) / 1048576.0;
        const double local_raw_mb = static_cast<double>(max_raw_bytes) / 1048576.0;
        const double local_comp_mb = static_cast<double>(max_comp_bytes) / 1048576.0;
        const double read_thr = agg.io_read_s > 0.0 ? raw_mb / agg.io_read_s : 0.0;
        const double comp_thr = agg.comp_s > 0.0 ? raw_mb / agg.comp_s : 0.0;
        const double write_thr = agg.io_write_s > 0.0 ? raw_mb / agg.io_write_s : 0.0;
        const double comp_write_s = agg.comp_s + agg.io_write_s;
        const double comp_write_thr = comp_write_s > 0.0 ? raw_mb / comp_write_s : 0.0;
        const double read_thr_local = agg.io_read_s > 0.0 ? local_raw_mb / agg.io_read_s : 0.0;
        const double comp_thr_local = agg.comp_s > 0.0 ? local_raw_mb / agg.comp_s : 0.0;
        const double write_thr_local = agg.io_write_s > 0.0 ? local_raw_mb / agg.io_write_s : 0.0;
        const double comp_write_thr_local = comp_write_s > 0.0 ? local_raw_mb / comp_write_s : 0.0;
        const double ratio = agg.raw_bytes > 0 ? static_cast<double>(agg.comp_bytes) / agg.raw_bytes : 0.0;

        std::cout << "\n[Summary]\n";
        std::cout << "  raw_read_s:     " << agg.io_read_s << " (" << read_thr << " MB/s)\n";
        std::cout << "  compress_s:     " << agg.comp_s << " (" << comp_thr << " MB/s)\n";
        std::cout << "  comp_write_s:   " << agg.io_write_s << " (" << write_thr << " MB/s)\n";
        std::cout << "  comp+write_s:   " << comp_write_s << " (" << comp_write_thr << " MB/s)\n";
        std::cout << "  raw_read_s_local:   " << agg.io_read_s << " (" << read_thr_local << " MB/s)\n";
        std::cout << "  compress_s_local:   " << agg.comp_s << " (" << comp_thr_local << " MB/s)\n";
        std::cout << "  comp_write_s_local: " << agg.io_write_s << " (" << write_thr_local << " MB/s)\n";
        std::cout << "  comp+write_s_local: " << comp_write_s << " (" << comp_write_thr_local << " MB/s)\n";
        std::cout << "  ratio:          " << ratio << " (comp/raw)\n";

        const std::string csv_path = opts.metrics_csv.empty()
            ? (opts.output_bin + ".mpi_metrics.csv")
            : opts.metrics_csv;
        append_metrics_csv(csv_path, opts, total_elems, elem_size_final, mpi.ranks, agg);
        std::cout << "  metrics_csv:    " << csv_path << "\n";
    }

    return 0;
}
