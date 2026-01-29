#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "mans_defs.h"

namespace mans::data_gen {

namespace fs = std::filesystem;

struct SyntheticConfig {
    double size_mb = 256.0;
    double ratio_smooth = 1.0;
    double ratio_spike = 0.0;
    double ratio_random = 0.0;
    int noise_range = 20;
    std::size_t block_size = 512;
    std::uint64_t seed = 42;
};

struct GeneratorConfig {
    SyntheticConfig synth{};
    std::string output_bin{};
    std::uint32_t dtype = mans::DataType::U32;
    std::uint32_t adm_threshold = 4000U;
};

inline std::string trim_copy(const std::string& str) {
    const char* whitespace = " \t\r\n";
    const auto first = str.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        return "";
    }
    const auto last = str.find_last_not_of(whitespace);
    return str.substr(first, last - first + 1);
}

inline void apply_config_kv(SyntheticConfig& cfg, const std::string& key, const std::string& val) {
    if (key == "size_mb") {
        cfg.size_mb = std::stod(val);
        return;
    }
    if (key == "ratio_smooth") {
        cfg.ratio_smooth = std::stod(val);
        return;
    }
    if (key == "ratio_spike") {
        cfg.ratio_spike = std::stod(val);
        return;
    }
    if (key == "ratio_random") {
        cfg.ratio_random = std::stod(val);
        return;
    }
    if (key == "noise_range") {
        cfg.noise_range = std::stoi(val);
        return;
    }
    if (key == "block_size") {
        cfg.block_size = static_cast<std::size_t>(std::stoull(val));
        return;
    }
    if (key == "seed") {
        cfg.seed = static_cast<std::uint64_t>(std::stoull(val));
        return;
    }
}

inline std::uint32_t parse_dtype_value(const std::string& val) {
    if (val == "u16" || val == "U16") {
        return mans::DataType::U16;
    }
    if (val == "u32" || val == "U32") {
        return mans::DataType::U32;
    }
    return static_cast<std::uint32_t>(std::stoul(val));
}

inline SyntheticConfig load_synthetic_config(const std::string& path) {
    SyntheticConfig cfg;
    if (path.empty()) {
        return cfg;
    }

    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open synthetic config: " + path);
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

        apply_config_kv(cfg, key, val);
    }

    return cfg;
}

inline GeneratorConfig load_generator_config(const std::string& path) {
    GeneratorConfig cfg;
    if (path.empty()) {
        return cfg;
    }

    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open generator config: " + path);
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

        if (key == "output_bin") {
            cfg.output_bin = val;
            continue;
        }
        if (key == "dtype") {
            cfg.dtype = parse_dtype_value(val);
            continue;
        }
        if (key == "adm_threshold") {
            cfg.adm_threshold = static_cast<std::uint32_t>(std::stoul(val));
            continue;
        }

        apply_config_kv(cfg.synth, key, val);
    }

    return cfg;
}

inline std::size_t aligned_total_elements(double size_mb, std::size_t elem_size, std::size_t block_size) {
    if (elem_size == 0) {
        throw std::runtime_error("Element size cannot be 0");
    }
    const auto total_bytes = static_cast<std::size_t>(size_mb * 1024.0 * 1024.0);
    auto elements = total_bytes / elem_size;
    if (elements == 0) {
        elements = block_size;
    }
    if (block_size == 0) {
        return elements;
    }
    const auto rem = elements % block_size;
    if (rem != 0) {
        elements += (block_size - rem);
    }
    return elements;
}

template <typename T>
inline std::size_t elements_from_file(const std::string& filename) {
    if (!fs::exists(filename)) {
        throw std::runtime_error("Input file not found: " + filename);
    }
    const auto bytes = fs::file_size(filename);
    if (bytes < sizeof(T)) {
        throw std::runtime_error("Input file too small for dtype");
    }
    return static_cast<std::size_t>(bytes / sizeof(T));
}

template <typename T>
inline std::vector<T> load_bin_slice(const std::string& filename,
                                     std::size_t offset_elems,
                                     std::size_t count_elems) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    const auto byte_offset = static_cast<std::uint64_t>(offset_elems) * sizeof(T);
    in.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
    if (!in.good()) {
        throw std::runtime_error("Failed to seek input file: " + filename);
    }

    std::vector<T> data(count_elems);
    const auto byte_count = static_cast<std::uint64_t>(count_elems) * sizeof(T);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(byte_count));
    const auto read_bytes = static_cast<std::uint64_t>(in.gcount());
    if (read_bytes != byte_count) {
        throw std::runtime_error("Short read from input file: " + filename);
    }

    return data;
}

template <typename T>
inline std::vector<T> generate_synthetic_slice(std::uint32_t adm_threshold,
                                               const SyntheticConfig& cfg,
                                               std::size_t total_elements,
                                               std::size_t slice_offset,
                                               std::size_t slice_count) {
    if (slice_count == 0) {
        return {};
    }
    if (cfg.block_size == 0) {
        throw std::runtime_error("Synthetic block_size cannot be 0");
    }

    const auto max_val = std::numeric_limits<T>::max();
    const std::vector<double> weights = {
        std::max(0.0, cfg.ratio_smooth),
        std::max(0.0, cfg.ratio_spike),
        std::max(0.0, cfg.ratio_random),
    };
    const double weight_sum = weights[0] + weights[1] + weights[2];
    if (weight_sum <= 0.0) {
        throw std::runtime_error("Synthetic ratios sum to 0");
    }

    std::vector<T> out(slice_count);

    const auto slice_end = slice_offset + slice_count;
    const auto first_block = slice_offset / cfg.block_size;
    const auto last_block = (slice_end - 1) / cfg.block_size;

    for (std::size_t block = first_block; block <= last_block; ++block) {
        const auto block_start = block * cfg.block_size;
        const auto block_end = std::min(block_start + cfg.block_size, total_elements);

        const auto overlap_start = std::max(block_start, slice_offset);
        const auto overlap_end = std::min(block_end, slice_end);
        if (overlap_start >= overlap_end) {
            continue;
        }

        const auto block_seed = cfg.seed + static_cast<std::uint64_t>(block * 1315423911ULL);
        std::mt19937_64 rng(block_seed);
        std::discrete_distribution<int> type_dist(weights.begin(), weights.end());
        std::uniform_int_distribution<int> noise_dist(0, std::max(0, cfg.noise_range));
        std::uniform_int_distribution<unsigned long long> full_range_dist(0, max_val);

        const int block_type = type_dist(rng);
        const T block_base = static_cast<T>(full_range_dist(rng));

        for (std::size_t i = overlap_start; i < overlap_end; ++i) {
            const auto local_idx = i - slice_offset;

            if (block_type == 0) {
                const int noise = noise_dist(rng);
                if (block_base > (max_val - static_cast<T>(std::max(0, cfg.noise_range)))) {
                    out[local_idx] = static_cast<T>(block_base - static_cast<T>(noise));
                } else {
                    out[local_idx] = static_cast<T>(block_base + static_cast<T>(noise));
                }
                continue;
            }

            if (block_type == 1) {
                out[local_idx] = block_base;
                const auto spike_pos = block_start + 1;
                if (i == spike_pos && spike_pos < block_end) {
                    const std::uint32_t spike_gap = adm_threshold + 500;
                    if (static_cast<unsigned long long>(block_base) + spike_gap > max_val) {
                        out[local_idx] = static_cast<T>(block_base - spike_gap);
                    } else {
                        out[local_idx] = static_cast<T>(block_base + spike_gap);
                    }
                }
                continue;
            }

            out[local_idx] = static_cast<T>(full_range_dist(rng));
        }
    }

    return out;
}

} // namespace mans::data_gen

namespace mans::h5 {
namespace data_gen = mans::data_gen;
} // namespace mans::h5

