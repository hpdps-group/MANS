#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hdf5.h>

#include "H5Z-MANS_config.h"
#include "mans_timing.h"
#include "mans_data_gen.h"
#include "../include/sz3_config_min.h"

#if defined(H5_HAVE_PARALLEL)
#include <mpi.h>
#endif

// ==========================================
// 1. Constants & Definitions
// ==========================================

// Filter IDs
#define FILTER_ID_NONE    0
#define FILTER_ID_DEFLATE 1
#define FILTER_ID_ZSTD    32015
#define FILTER_ID_MANS    32001
#define FILTER_ID_SZ3     32024

// Synthetic defaults (kept aligned with previous hard-coded values)
static constexpr double DEFAULT_DATA_SIZE_MB = 256.0;
static constexpr double DEFAULT_CHUNK_MB = 32.0;

// Color Definitions
static const std::string RST  = "\033[0m";
static const std::string RED  = "\033[1;31m";
static const std::string GRN  = "\033[1;32m";
static const std::string YLW  = "\033[1;33m";
static const std::string BLU  = "\033[1;36m";
static const std::string BOLD = "\033[1m";

// ==========================================
// 2. Runtime Configuration Structs
// ==========================================

enum class Mode {
    Compress,
    Decompress,
};

struct RunOptions {
    std::string input_bin;
    std::string output_h5;

    double chunk_mb = DEFAULT_CHUNK_MB;
    int filter_id = FILTER_ID_MANS;
    Mode mode = Mode::Compress;

    // Optional: validate against provided rank count.
    int expected_ranks = -1;

    // Synthetic data config (used only when input_bin is empty).
    mans::h5::data_gen::SyntheticConfig synth_cfg{};

    // Optional: point to a dataset generator config file.
    std::string dataset_config_file;

    // MANS config is only required when filter==MANS.
    std::string mans_config_file;

    // Output metrics CSV. If empty, we derive from output_h5.
    std::string metrics_csv;
};

struct Metrics {
    double raw_io_s = 0.0;        // compress: raw read; decompress: raw write
    double h5_io_s = 0.0;         // compress: H5Dwrite; decompress: H5Dread
    std::uint64_t raw_bytes = 0;  // raw byte count processed by this rank
};

// ==========================================
// 3. Helper Functions
// ==========================================

#define CHECK_H5(func_call)                                                            \
    do {                                                                               \
        if ((func_call) < 0) {                                                         \
            std::cerr << RED << "[Error] HDF5 call failed at line " << __LINE__       \
                      << ": " #func_call << RST << "\n";                           \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

namespace fs = std::filesystem;

static std::string mode_to_string(Mode mode) {
    return mode == Mode::Compress ? "compress" : "decompress";
}

static std::string filter_to_string(int filter_id) {
    switch (filter_id) {
        case FILTER_ID_NONE:
            return "none";
        case FILTER_ID_DEFLATE:
            return "gzip";
        case FILTER_ID_ZSTD:
            return "zstd";
        case FILTER_ID_MANS:
            return "mans";
        case FILTER_ID_SZ3:
            return "sz3";
        default:
            return std::to_string(filter_id);
    }
}

static std::size_t safe_chunk_elems(double chunk_mb, std::size_t elem_size, std::size_t total_elems) {
    const auto chunk_bytes = static_cast<std::size_t>(chunk_mb * 1024.0 * 1024.0);
    std::size_t chunk_elems = elem_size == 0 ? 0 : (chunk_bytes / elem_size);
    if (chunk_elems == 0) {
        chunk_elems = 1;
    }
    if (total_elems > 0) {
        chunk_elems = std::min<std::size_t>(chunk_elems, total_elems);
    }
    return chunk_elems;
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

static double seconds_since(const std::chrono::steady_clock::time_point& t0,
                            const std::chrono::steady_clock::time_point& t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

static void append_metrics_csv(const std::string& csv_path,
                               const RunOptions& opts,
                               std::size_t total_elems,
                               std::size_t elem_size,
                               int ranks,
                               const Metrics& agg,
                               double raw_thr_mb_s,
                               double h5_thr_mb_s) {
    const bool exists = fs::exists(csv_path);
    std::ofstream out(csv_path, std::ios::app);
    if (!out.is_open()) {
        std::cerr << YLW << "[WARN] Failed to open metrics CSV: " << csv_path << RST << "\n";
        return;
    }

    if (!exists) {
        out << "mode,filter,chunk_mb,ranks,total_elems,elem_size_bytes,raw_bytes,"
               "raw_io_s,raw_thr_mb_s,h5_io_s,h5_thr_mb_s\n";
    }

    out << mode_to_string(opts.mode) << ","
        << filter_to_string(opts.filter_id) << ","
        << opts.chunk_mb << ","
        << ranks << ","
        << total_elems << ","
        << elem_size << ","
        << agg.raw_bytes << ","
        << std::fixed << std::setprecision(6)
        << agg.raw_io_s << ","
        << raw_thr_mb_s << ","
        << agg.h5_io_s << ","
        << h5_thr_mb_s
        << "\n";
}

#if defined(H5_HAVE_PARALLEL)
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
#else
struct MpiContext {
    int rank = 0;
    int ranks = 1;
    bool initialized = false;

    MpiContext(int&, char**&) {}
};
#endif

// ==========================================
// 4. Core MPI Tester
// ==========================================

template <typename T>
class MpiTester {
public:
    MpiTester(const mans::h5::MansConfig* mans_cfg,
              hid_t h5_type,
              const RunOptions& opts,
              int rank,
              int ranks)
        : mans_config_(mans_cfg),
          h5_native_type_(h5_type),
          options_(opts),
          rank_(rank),
          ranks_(ranks) {}

    int run() {
        try {
            setup_totals();
            print_banner();

            Metrics local;
            if (options_.mode == Mode::Compress) {
                local = compress_write();
            } else {
                local = decompress_read();
            }

            finalize_and_report(local);
        } catch (const std::exception& e) {
            std::cerr << RED << "[Rank " << rank_ << "] Exception: " << e.what() << RST << "\n";
            return 1;
        }
        return 0;
    }

private:
    const mans::h5::MansConfig* mans_config_ = nullptr;
    hid_t h5_native_type_ = H5T_NATIVE_UINT32;
    RunOptions options_{};

    int rank_ = 0;
    int ranks_ = 1;

    mans::h5::data_gen::SyntheticConfig synth_cfg_{};
    std::size_t total_elements_ = 0;
    std::size_t elem_size_ = sizeof(T);

    std::size_t rank_offset_ = 0;
    std::size_t rank_count_ = 0;

    std::size_t chunk_elems_ = 0;

    void setup_totals() {
        synth_cfg_ = options_.synth_cfg;
        const bool has_input = !options_.input_bin.empty();

        if (!has_input) {
            if (rank_ == 0) {
                std::cerr << YLW
                          << "[WARN] No input_bin provided. Falling back to synthetic data generated in memory only. "
                             "Use mans_data_gen to control and persist datasets."
                          << RST << "\n";
            }
            if (synth_cfg_.size_mb <= 0.0) {
                synth_cfg_.size_mb = DEFAULT_DATA_SIZE_MB;
                if (rank_ == 0) {
                    std::cerr << YLW << "[WARN] Synthetic size_mb not set. Using default "
                              << synth_cfg_.size_mb << " MB." << RST << "\n";
                }
            }
        }

        if (has_input) {
            if (!fs::exists(options_.input_bin)) {
                throw std::runtime_error("input_bin not found: " + options_.input_bin);
            }
            const auto bytes = fs::file_size(options_.input_bin);
            if (bytes < elem_size_) {
                throw std::runtime_error("input_bin is smaller than one element for the selected dtype");
            }
            if (rank_ == 0 && (bytes % elem_size_) != 0) {
                std::cerr << YLW
                          << "[WARN] input_bin size is not a multiple of element size. Trailing bytes will be ignored."
                          << RST << "\n";
            }
            total_elements_ = static_cast<std::size_t>(bytes / elem_size_);
        } else {
            total_elements_ = mans::h5::data_gen::aligned_total_elements(
                synth_cfg_.size_mb,
                elem_size_,
                synth_cfg_.block_size);
        }

        if (total_elements_ == 0) {
            throw std::runtime_error("Computed total_elements is 0");
        }

        split_even(total_elements_, rank_, ranks_, rank_offset_, rank_count_);
        chunk_elems_ = safe_chunk_elems(options_.chunk_mb, elem_size_, total_elements_);

        if (rank_ == 0 && chunk_elems_ >= total_elements_) {
            std::cerr << YLW << "[WARN] chunk size covers the entire dataset; only one chunk will be used." << RST
                      << "\n";
        }
    }

    void print_banner() const {
        if (rank_ != 0) {
            return;
        }

        std::cout << BOLD
                  << "========================================\n"
                  << "   H5Z-MANS MPI Test Runner (Chunked)   \n"
                  << "========================================" << RST << "\n";
        std::cout << " Mode:    " << mode_to_string(options_.mode) << "\n";
        std::cout << " Filter:  " << filter_to_string(options_.filter_id) << " (" << options_.filter_id << ")\n";
        std::cout << " Ranks:   " << ranks_ << "\n";
        std::cout << " Output:  " << options_.output_h5 << "\n";
        std::cout << " Input:   " << (options_.input_bin.empty() ? "<synthetic>" : options_.input_bin) << "\n";
        std::cout << " Total:   " << total_elements_ << " elems (" << (total_elements_ * elem_size_) / 1048576.0
                  << " MB raw)\n";
        std::cout << " Chunk:   " << options_.chunk_mb << " MB (" << chunk_elems_ << " elems)\n";
        if (options_.filter_id == FILTER_ID_MANS && !options_.mans_config_file.empty()) {
            std::cout << " MANS:    " << options_.mans_config_file << "\n";
        }
        std::cout << "\n";
    }

    hid_t create_file_access_plist() const {
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        CHECK_H5(fapl);

#if defined(H5_HAVE_PARALLEL)
        CHECK_H5(H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL));
#else
        if (ranks_ > 1 && rank_ == 0) {
            std::cerr << YLW << "[WARN] HDF5 is not parallel-enabled; running serial access." << RST << "\n";
        }
#endif
        return fapl;
    }

    hid_t create_dxpl_collective() const {
        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        CHECK_H5(dxpl);
#if defined(H5_HAVE_PARALLEL)
        CHECK_H5(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE));
#endif
        return dxpl;
    }

    void configure_filter(hid_t dcpl) const {
        const hsize_t chunk[1] = {static_cast<hsize_t>(chunk_elems_)};
        CHECK_H5(H5Pset_chunk(dcpl, 1, chunk));

        if (options_.filter_id == FILTER_ID_NONE) {
            if (rank_ == 0) {
                std::cout << "  -> Filter: " << BLU << "None" << RST << "\n";
            }
            return;
        }

        if (options_.filter_id == FILTER_ID_DEFLATE) {
            const unsigned int gzip_level = 6;
            if (rank_ == 0) {
                std::cout << "  -> Filter: " << BLU << "GZIP (Level " << gzip_level << ")" << RST << "\n";
            }
            CHECK_H5(H5Pset_deflate(dcpl, gzip_level));
            return;
        }

        if (options_.filter_id == FILTER_ID_ZSTD) {
            const unsigned int zstd_level = 3;
            if (rank_ == 0) {
                std::cout << "  -> Filter: " << BLU << "Zstandard (Level " << zstd_level << ")" << RST << "\n";
            }
            const htri_t avail = H5Zfilter_avail(FILTER_ID_ZSTD);
            if (!avail) {
                throw std::runtime_error("Zstandard filter not available");
            }
            unsigned int cd_values[1] = {zstd_level};
            CHECK_H5(H5Pset_filter(dcpl, FILTER_ID_ZSTD, H5Z_FLAG_OPTIONAL, 1, cd_values));
            return;
        }

        if (options_.filter_id == FILTER_ID_MANS) {
            if (!mans_config_) {
                throw std::runtime_error("MANS filter requires --mans-config <file>");
            }
            if (rank_ == 0) {
                std::cout << "  -> Filter: " << BLU << "MANS Custom (ID " << FILTER_ID_MANS << ")" << RST << "\n";
            }
            const auto cd = mans_config_->to_cd_values();
            CHECK_H5(H5Pset_filter(dcpl, FILTER_ID_MANS, 0, cd.size(), cd.data()));
            return;
        }

        if (options_.filter_id == FILTER_ID_SZ3) {
            if (rank_ == 0) {
                std::cout << "  -> Filter: " << BLU << "SZ3 Compressor (ID " << FILTER_ID_SZ3 << ")" << RST << "\n";
            }
            const htri_t avail = H5Zfilter_avail(FILTER_ID_SZ3);
            if (!avail) {
                throw std::runtime_error("SZ3 filter not available");
            }

            SZ3::Config sz3_conf;
            static const char* sz3_ini = R"ini([GlobalSettings]
CmprAlgo = ALGO_INTERP_LORENZO
ErrorBoundMode = ABS
AbsErrorBound = 1e-3
OpenMP = YES

[AlgoSettings]
)ini";
            sz3_conf.load_ini(sz3_ini);

            std::size_t cd_nelmts = static_cast<std::size_t>(
                std::ceil(sz3_conf.size_est() / static_cast<double>(sizeof(int))));
            std::vector<unsigned int> cd_values(cd_nelmts, 0);
            auto* buffer = reinterpret_cast<unsigned char*>(cd_values.data());
            const auto conf_size_real = sz3_conf.save(buffer);
            cd_nelmts = static_cast<std::size_t>(
                std::ceil(conf_size_real / static_cast<double>(sizeof(int))));

            CHECK_H5(H5Pset_filter(dcpl, FILTER_ID_SZ3, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values.data()));
            return;
        }

        throw std::runtime_error("Unknown filter id: " + std::to_string(options_.filter_id));
    }

    std::vector<T> load_or_generate_chunk(std::size_t global_offset, std::size_t count) const {
        if (count == 0) {
            return {};
        }

        if (!options_.input_bin.empty()) {
            return mans::h5::data_gen::load_bin_slice<T>(options_.input_bin, global_offset, count);
        }

        const std::uint32_t threshold = mans_config_ ? mans_config_->get_params().adm_threshold : 4000U;
        return mans::h5::data_gen::generate_synthetic_slice<T>(
            threshold,
            synth_cfg_,
            total_elements_,
            global_offset,
            count);
    }

    Metrics compress_write() {
        if (rank_ == 0) {
            std::cout << BOLD << "[Compress] Writing chunked dataset..." << RST << "\n";
        }

        const hid_t fapl = create_file_access_plist();
        const hid_t file = H5Fcreate(options_.output_h5.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        CHECK_H5(file);
        CHECK_H5(H5Pclose(fapl));

        const hsize_t dims[1] = {static_cast<hsize_t>(total_elements_)};
        const hid_t filespace = H5Screate_simple(1, dims, nullptr);
        CHECK_H5(filespace);

        const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        CHECK_H5(dcpl);
        configure_filter(dcpl);

        const hid_t dset = H5Dcreate2(file, "dataset", h5_native_type_, filespace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        CHECK_H5(dset);
        CHECK_H5(H5Pclose(dcpl));
        CHECK_H5(H5Sclose(filespace));

        const hid_t dxpl = create_dxpl_collective();

        Metrics m{};
        T dummy_value{};

        const hid_t fspace = H5Dget_space(dset);
        CHECK_H5(fspace);

        if (rank_count_ > 0) {
            const hsize_t file_offset[1] = {static_cast<hsize_t>(rank_offset_)};
            const hsize_t file_count[1] = {static_cast<hsize_t>(rank_count_)};
            CHECK_H5(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, file_offset, nullptr, file_count, nullptr));

            auto t0 = std::chrono::steady_clock::now();
            std::vector<T> slice = load_or_generate_chunk(rank_offset_, rank_count_);
            auto t1 = std::chrono::steady_clock::now();
            m.raw_io_s += seconds_since(t0, t1);
            m.raw_bytes += static_cast<std::uint64_t>(rank_count_) * elem_size_;

            const hid_t mspace = H5Screate_simple(1, file_count, nullptr);
            CHECK_H5(mspace);

            auto t2 = std::chrono::steady_clock::now();
            {
                MANS_TIMING_RUN_SCOPE();
                CHECK_H5(H5Dwrite(dset, h5_native_type_, mspace, fspace, dxpl, slice.data()));
            }
            auto t3 = std::chrono::steady_clock::now();
            m.h5_io_s += seconds_since(t2, t3);

            CHECK_H5(H5Sclose(mspace));
        } else {
            CHECK_H5(H5Sselect_none(fspace));
            const hsize_t one[1] = {1};
            const hid_t mspace = H5Screate_simple(1, one, nullptr);
            CHECK_H5(mspace);
            CHECK_H5(H5Sselect_none(mspace));

            {
                MANS_TIMING_RUN_SCOPE();
                CHECK_H5(H5Dwrite(dset, h5_native_type_, mspace, fspace, dxpl, &dummy_value));
            }

            CHECK_H5(H5Sclose(mspace));
        }

        CHECK_H5(H5Sclose(fspace));

        CHECK_H5(H5Fflush(file, H5F_SCOPE_GLOBAL));
        CHECK_H5(H5Pclose(dxpl));
        CHECK_H5(H5Dclose(dset));
        CHECK_H5(H5Fclose(file));

        return m;
    }

    Metrics decompress_read() {
        if (rank_ == 0) {
            std::cout << BOLD << "[Decompress] Reading chunked dataset..." << RST << "\n";
        }

        const hid_t fapl = create_file_access_plist();
        const hid_t file = H5Fopen(options_.output_h5.c_str(), H5F_ACC_RDONLY, fapl);
        CHECK_H5(file);
        CHECK_H5(H5Pclose(fapl));

        const hid_t dset = H5Dopen2(file, "dataset", H5P_DEFAULT);
        CHECK_H5(dset);

        const hid_t dxpl = create_dxpl_collective();

        Metrics m{};
        T dummy_value{};

        const hid_t fspace = H5Dget_space(dset);
        CHECK_H5(fspace);

        if (rank_count_ > 0) {
            const hsize_t file_offset[1] = {static_cast<hsize_t>(rank_offset_)};
            const hsize_t file_count[1] = {static_cast<hsize_t>(rank_count_)};
            CHECK_H5(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, file_offset, nullptr, file_count, nullptr));

            std::vector<T> slice(rank_count_);
            m.raw_bytes += static_cast<std::uint64_t>(rank_count_) * elem_size_;

            const hid_t mspace = H5Screate_simple(1, file_count, nullptr);
            CHECK_H5(mspace);

            auto t0 = std::chrono::steady_clock::now();
            {
                MANS_TIMING_RUN_SCOPE();
                CHECK_H5(H5Dread(dset, h5_native_type_, mspace, fspace, dxpl, slice.data()));
            }
            auto t1 = std::chrono::steady_clock::now();
            m.h5_io_s += seconds_since(t0, t1);

            auto t2 = std::chrono::steady_clock::now();
            const std::string dump_path = options_.output_h5 + ".rank" + std::to_string(rank_) + ".raw.bin";
            {
                std::ofstream out(dump_path, std::ios::binary | std::ios::app);
                if (!out.is_open()) {
                    throw std::runtime_error("Failed to open raw dump: " + dump_path);
                }
                out.write(reinterpret_cast<const char*>(slice.data()),
                          static_cast<std::streamsize>(slice.size() * sizeof(T)));
            }
            auto t3 = std::chrono::steady_clock::now();
            m.raw_io_s += seconds_since(t2, t3);

            CHECK_H5(H5Sclose(mspace));
        } else {
            CHECK_H5(H5Sselect_none(fspace));
            const hsize_t one[1] = {1};
            const hid_t mspace = H5Screate_simple(1, one, nullptr);
            CHECK_H5(mspace);
            CHECK_H5(H5Sselect_none(mspace));

            {
                MANS_TIMING_RUN_SCOPE();
                CHECK_H5(H5Dread(dset, h5_native_type_, mspace, fspace, dxpl, &dummy_value));
            }

            CHECK_H5(H5Sclose(mspace));
        }

        CHECK_H5(H5Sclose(fspace));

        CHECK_H5(H5Pclose(dxpl));
        CHECK_H5(H5Dclose(dset));
        CHECK_H5(H5Fclose(file));

        return m;
    }

    void finalize_and_report(const Metrics& local) const {
        Metrics agg = local;

#if defined(H5_HAVE_PARALLEL)
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

        agg.raw_io_s = reduce_max(local.raw_io_s);
        agg.h5_io_s = reduce_max(local.h5_io_s);
        agg.raw_bytes = reduce_sum_u64(local.raw_bytes);
#else
        (void)local;
#endif

        if (rank_ != 0) {
            return;
        }

        const double raw_mb = static_cast<double>(agg.raw_bytes) / 1048576.0;
        const double raw_io_thr = agg.raw_io_s > 0.0 ? raw_mb / agg.raw_io_s : 0.0;
        const double h5_thr = agg.h5_io_s > 0.0 ? raw_mb / agg.h5_io_s : 0.0;

        std::cout << "\n" << BOLD << "[Summary]" << RST << "\n";
        if (options_.mode == Mode::Compress) {
            std::cout << "  raw_read_s:     " << BLU << agg.raw_io_s << RST << " (" << BLU << raw_io_thr << " MB/s" << RST << ")\n";
            std::cout << "  h5dwrite_s:     " << BLU << agg.h5_io_s << RST << " (" << BLU << h5_thr << " MB/s" << RST << ")\n";
        } else {
            std::cout << "  h5dread_s:      " << BLU << agg.h5_io_s << RST << " (" << BLU << h5_thr << " MB/s" << RST << ")\n";
            std::cout << "  raw_write_s:    " << BLU << agg.raw_io_s << RST << " (" << BLU << raw_io_thr << " MB/s" << RST << ")\n";
        }

        const std::string csv_path = options_.metrics_csv.empty()
            ? (options_.output_h5 + ".mpi_metrics.csv")
            : options_.metrics_csv;
        append_metrics_csv(csv_path, options_, total_elements_, elem_size_, ranks_, agg, raw_io_thr, h5_thr);

        std::cout << "  metrics_csv:    " << GRN << csv_path << RST << "\n";
    }
};

// ==========================================
// 5. CLI Parsing
// ==========================================

static void print_usage(const char* prog) {
    std::cerr << YLW
              << "Usage:\n  " << prog
              << " --config <test.cfg> [--input raw.bin] [--output out.h5] [--chunk-mb MB]\\\n"
                 "         [--filter mans|zstd|gzip|sz3|none] [--mode compress|decompress]\\\n"
                 "         [--ranks N] [--mans-config mans.cfg] [--dataset-config synth.cfg] [--csv metrics.csv]"
              << RST << "\n";
}

static int parse_filter(std::string_view name) {
    if (name == "none") {
        return FILTER_ID_NONE;
    }
    if (name == "gzip" || name == "deflate") {
        return FILTER_ID_DEFLATE;
    }
    if (name == "zstd") {
        return FILTER_ID_ZSTD;
    }
    if (name == "mans") {
        return FILTER_ID_MANS;
    }
    if (name == "sz3") {
        return FILTER_ID_SZ3;
    }
    return std::stoi(std::string(name));
}

static Mode parse_mode(std::string_view name) {
    if (name == "compress") {
        return Mode::Compress;
    }
    if (name == "decompress") {
        return Mode::Decompress;
    }
    throw std::runtime_error("Unknown mode: " + std::string(name));
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

static bool is_dataset_config_key(const std::string& key) {
    return key == "size_mb" || key == "ratio_smooth" || key == "ratio_spike" || key == "ratio_random" ||
           key == "noise_range" || key == "block_size" || key == "seed" ||
           key == "output_bin" || key == "dtype" || key == "adm_threshold";
}

static void apply_run_config_kv(RunOptions& opts,
                                const std::string& key,
                                const std::string& val,
                                int rank) {
    if (key == "input" || key == "input_bin") {
        opts.input_bin = val;
        return;
    }
    if (key == "output" || key == "output_h5") {
        opts.output_h5 = val;
        return;
    }
    if (key == "chunk" || key == "chunk_mb") {
        opts.chunk_mb = std::stod(val);
        return;
    }
    if (key == "filter") {
        opts.filter_id = parse_filter(val);
        return;
    }
    if (key == "mode") {
        opts.mode = parse_mode(val);
        return;
    }
    if (key == "ranks" || key == "expected_ranks") {
        opts.expected_ranks = std::stoi(val);
        return;
    }
    if (key == "mans_config" || key == "mans_config_file") {
        opts.mans_config_file = val;
        return;
    }
    if (key == "dataset_config" || key == "dataset_config_file") {
        opts.dataset_config_file = val;
        return;
    }
    if (key == "csv" || key == "metrics_csv") {
        opts.metrics_csv = val;
        return;
    }

    if (is_dataset_config_key(key)) {
        if (rank == 0) {
            std::cerr << YLW << "[WARN] Dataset generation key '" << key
                      << "' is ignored by H5Z-MANS_test. Use mans_data_gen instead." << RST << "\n";
        }
        return;
    }

    if (rank == 0) {
        std::cerr << YLW << "[WARN] Unknown config key: " << key << RST << "\n";
    }
}

static void load_run_config(const std::string& path, RunOptions& opts, int rank) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open config: " + path);
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
        apply_run_config_kv(opts, key, val, rank);
    }
}

static std::optional<RunOptions> parse_args(int argc, char** argv, int rank) {
    RunOptions opts;
    std::string config_file;

    // First pass: find --config so it can provide defaults.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg != "--config") {
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for --config");
        }
        config_file = argv[++i];
    }

    if (!config_file.empty()) {
        load_run_config(config_file, opts, rank);
    }

    // Second pass: CLI overrides.
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
            opts.output_h5 = need_value("--output");
            continue;
        }
        if (arg == "--chunk-mb") {
            opts.chunk_mb = std::stod(need_value("--chunk-mb"));
            continue;
        }
        if (arg == "--filter") {
            opts.filter_id = parse_filter(need_value("--filter"));
            continue;
        }
        if (arg == "--mode") {
            opts.mode = parse_mode(need_value("--mode"));
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
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return std::nullopt;
        }

        throw std::runtime_error("Unknown argument: " + arg);
    }

    // Load dataset config after CLI overrides so it behaves like a default.
    if (opts.input_bin.empty() && !opts.dataset_config_file.empty()) {
        const auto gen_cfg = mans::h5::data_gen::load_generator_config(opts.dataset_config_file);
        opts.synth_cfg = gen_cfg.synth;
        if (rank == 0 && (!gen_cfg.output_bin.empty() || gen_cfg.dtype != mans::DataType::U32 ||
                          gen_cfg.adm_threshold != 4000U)) {
            std::cerr << YLW
                      << "[WARN] dataset_config_file provides output/dtype/adm_threshold which are ignored by "
                         "H5Z-MANS_test."
                      << RST << "\n";
        }
    }

    if (opts.output_h5.empty() || opts.chunk_mb <= 0.0) {
        print_usage(argv[0]);
        return std::nullopt;
    }

    // Keep previous defaults where not overridden.
    if (opts.filter_id == FILTER_ID_MANS && opts.mans_config_file.empty()) {
        std::cerr << RED << "[Error] --mans-config is required when --filter mans" << RST << "\n";
        return std::nullopt;
    }

    return opts;
}

// ==========================================
// 6. Main
// ==========================================

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
            std::cerr << RED << "Arg parse error: " << e.what() << RST << "\n";
            print_usage(argv[0]);
        }
        return 1;
    }

    if (opts.expected_ranks > 0 && opts.expected_ranks != mpi.ranks && mpi.rank == 0) {
        std::cerr << YLW << "[WARN] --ranks " << opts.expected_ranks
                  << " does not match MPI world size " << mpi.ranks << ". Using MPI size." << RST << "\n";
    }

    std::optional<mans::h5::MansConfig> mans_cfg;
    if (opts.filter_id == FILTER_ID_MANS) {
        mans_cfg.emplace();
        try {
            mans_cfg->load(opts.mans_config_file);
        } catch (const std::exception& e) {
            if (mpi.rank == 0) {
                std::cerr << RED << "Config Error: " << e.what() << RST << "\n";
            }
            return 1;
        }
    }

    std::uint32_t data_type = mans::DataType::U32;
    if (mans_cfg.has_value()) {
        data_type = mans_cfg->get_params().dtype;
    }

    int rc = 0;
    if (data_type == mans::DataType::U16) {
        MpiTester<std::uint16_t> tester(mans_cfg ? &*mans_cfg : nullptr, H5T_NATIVE_UINT16, opts, mpi.rank, mpi.ranks);
        rc = tester.run();
    } else {
        MpiTester<std::uint32_t> tester(mans_cfg ? &*mans_cfg : nullptr, H5T_NATIVE_UINT32, opts, mpi.rank, mpi.ranks);
        rc = tester.run();
    }

    if (0 > H5close()) {
        if (mpi.rank == 0) {
            std::cerr << RED << "Error in H5close" << RST << "\n";
        }
        return 1;
    }

    if (mpi.rank == 0) {
        std::cout << "Done\n";
    }
    return rc;
}
