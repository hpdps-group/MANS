#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "H5Z-MANS_config.h"
#include "mans_data_gen.h"

namespace fs = std::filesystem;

struct GenOptions {
    std::string output_bin;
    std::string config_file;
    std::string synth_config_file;
    std::string mans_config_file;

    std::optional<double> size_mb_override;
    std::optional<std::uint32_t> adm_threshold_override;
    std::optional<std::uint32_t> dtype_override;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage:\n  " << prog
        << " [--config gen.cfg] [--output data.bin] [--synth-config synth.cfg] [--size-mb MB]\\\n"
           "         [--dtype u16|u32] [--adm-threshold N] [--mans-config mans.cfg]\n";
}

static std::uint32_t parse_dtype(std::string_view s) {
    if (s == "u16" || s == "U16") {
        return mans::DataType::U16;
    }
    if (s == "u32" || s == "U32") {
        return mans::DataType::U32;
    }
    return static_cast<std::uint32_t>(std::stoul(std::string(s)));
}

static std::optional<GenOptions> parse_args(int argc, char** argv) {
    GenOptions opts;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--output") {
            opts.output_bin = need_value("--output");
            continue;
        }
        if (arg == "--config") {
            opts.config_file = need_value("--config");
            continue;
        }
        if (arg == "--synth-config") {
            opts.synth_config_file = need_value("--synth-config");
            continue;
        }
        if (arg == "--size-mb") {
            opts.size_mb_override = std::stod(need_value("--size-mb"));
            continue;
        }
        if (arg == "--dtype") {
            opts.dtype_override = parse_dtype(need_value("--dtype"));
            continue;
        }
        if (arg == "--adm-threshold") {
            opts.adm_threshold_override = static_cast<std::uint32_t>(
                std::stoul(need_value("--adm-threshold")));
            continue;
        }
        if (arg == "--mans-config") {
            opts.mans_config_file = need_value("--mans-config");
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return std::nullopt;
        }

        throw std::runtime_error("Unknown argument: " + arg);
    }

    return opts;
}

template <typename T>
static void write_bin(const std::string& path, const std::vector<T>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output: " + path);
    }
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(T)));
}

template <typename T>
static int generate_and_write(const mans::h5::data_gen::SyntheticConfig& cfg,
                              std::uint32_t adm_threshold,
                              const std::string& output_path) {
    const std::size_t elem_size = sizeof(T);
    const std::size_t total_elements = mans::h5::data_gen::aligned_total_elements(
        cfg.size_mb,
        elem_size,
        cfg.block_size);

    const auto data = mans::h5::data_gen::generate_synthetic_slice<T>(
        adm_threshold,
        cfg,
        total_elements,
        /*slice_offset=*/0,
        /*slice_count=*/total_elements);

    write_bin(output_path, data);

    const double raw_mb = (static_cast<double>(total_elements) * elem_size) / 1048576.0;
    std::cout << "Generated: " << output_path << "\n";
    std::cout << "  dtype:        " << (elem_size == 2 ? "u16" : "u32") << "\n";
    std::cout << "  elements:     " << total_elements << "\n";
    std::cout << "  raw_size_mb:  " << raw_mb << "\n";
    std::cout << "  adm_threshold:" << adm_threshold << "\n";
    return 0;
}

int main(int argc, char** argv) {
    GenOptions opts;
    try {
        const auto parsed = parse_args(argc, argv);
        if (!parsed.has_value()) {
            return 1;
        }
        opts = *parsed;
    } catch (const std::exception& e) {
        std::cerr << "Arg parse error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    mans::h5::data_gen::GeneratorConfig gen_cfg;
    try {
        if (!opts.config_file.empty()) {
            gen_cfg = mans::h5::data_gen::load_generator_config(opts.config_file);
        } else {
            gen_cfg.synth = mans::h5::data_gen::load_synthetic_config(opts.synth_config_file);
        }
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    if (opts.size_mb_override.has_value()) {
        gen_cfg.synth.size_mb = *opts.size_mb_override;
    }
    if (gen_cfg.synth.size_mb <= 0.0) {
        gen_cfg.synth.size_mb = 256.0;
    }

    std::uint32_t dtype = gen_cfg.dtype;
    std::uint32_t adm_threshold = gen_cfg.adm_threshold;

    std::optional<mans::h5::MansConfig> mans_cfg;
    if (!opts.mans_config_file.empty()) {
        mans_cfg.emplace();
        try {
            mans_cfg->load(opts.mans_config_file);
        } catch (const std::exception& e) {
            std::cerr << "MANS config error: " << e.what() << "\n";
            return 1;
        }
        dtype = mans_cfg->get_params().dtype;
        adm_threshold = mans_cfg->get_params().adm_threshold;
    }

    if (opts.dtype_override.has_value()) {
        dtype = *opts.dtype_override;
    }
    if (opts.adm_threshold_override.has_value()) {
        adm_threshold = *opts.adm_threshold_override;
    }

    const std::string output_path = !opts.output_bin.empty() ? opts.output_bin : gen_cfg.output_bin;
    if (output_path.empty()) {
        std::cerr << "Missing output path. Provide --output or set output_bin in --config.\n";
        print_usage(argv[0]);
        return 1;
    }

    try {
        if (dtype == mans::DataType::U16) {
            return generate_and_write<std::uint16_t>(gen_cfg.synth, adm_threshold, output_path);
        }
        return generate_and_write<std::uint32_t>(gen_cfg.synth, adm_threshold, output_path);
    } catch (const std::exception& e) {
        std::cerr << "Generation error: " << e.what() << "\n";
        return 1;
    }
}
