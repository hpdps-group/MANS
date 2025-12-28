#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <chrono>

#include <hdf5.h>
#include "H5Z-MANS_config.h" 

// ==========================================
// Color Definitions (ANSI Escape Codes)
// ==========================================
const std::string RST  = "\033[0m";      // Reset
const std::string RED  = "\033[1;31m";   // Error / Failure
const std::string GRN  = "\033[1;32m";   // Success
const std::string YLW  = "\033[1;33m";   // Warning / Fallback
const std::string BLU  = "\033[1;36m";   // Metrics / Info (Cyan for better visibility)
const std::string BOLD = "\033[1m";      // Titles

// ==========================================
// Global Helper Definitions
// ==========================================

#define H5Z_FILTER_MANS_ID 32001

// Macro with Red Color for Errors
#define CHECK_H5(func_call) \
    do { \
        if ((func_call) < 0) { \
            std::cerr << RED << "[Error] HDF5 call failed at line " << __LINE__ << ": " #func_call << RST << "\n"; \
            std::exit(1); \
        } \
    } while (0)

namespace fs = std::filesystem;

// Get file size
size_t get_file_size(const std::string& filename) {
    if (fs::exists(filename)) return fs::file_size(filename);
    return 0;
}

// Generic file write
template <typename T>
void save_bin(const std::string& filename, const std::vector<T>& data) {
    std::ofstream outfile(filename, std::ios::binary);
    if (outfile) {
        outfile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
        std::cout << "  -> Saved dump: " << GRN << filename << RST << "\n";
    }
}

// Generic file read
template <typename T>
std::vector<T> load_bin(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary | std::ios::ate);
    if (!infile) throw std::runtime_error("Cannot open file: " + filename);
    size_t bytes = infile.tellg();
    infile.seekg(0);
    std::vector<T> data(bytes / sizeof(T));
    infile.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

// Generate synthetic test data (Smooth + Spikes)
template <typename T>
std::vector<T> generate_synthetic_data(size_t num_elements, uint32_t threshold) {
    std::vector<T> data(num_elements);
    std::mt19937 rng(42);
    size_t block_size = 512;
    size_t num_blocks = num_elements / block_size;
    if (num_blocks == 0) num_blocks = 1;

    for (size_t b = 0; b < num_blocks; ++b) {
        size_t start = b * block_size;
        size_t end = std::min(start + block_size, num_elements);
        bool trigger_adm = (b % 2 == 0);

        if (trigger_adm) {
            T base = 10000;
            uint32_t range = (threshold > 10) ? (threshold / 2) : 1;
            std::uniform_int_distribution<int> dist(0, range);
            for (size_t i = start; i < end; ++i) data[i] = static_cast<T>(base + dist(rng));
        } else {
            T base = 20000;
            for (size_t i = start; i < end; ++i) data[i] = static_cast<T>(base);
            if (start + 1 < end) {
                // Create spikes
                uint32_t spike = base + threshold + 500;
                data[start + 1] = static_cast<T>((sizeof(T) == 2 && spike > 65535) ? (base - threshold - 500) : spike);
            }
        }
    }
    return data;
}

// ==========================================
// Core Tester Class
// ==========================================
template <typename T>
class Tester {
public:
    Tester(const mans::h5::MansConfig& cfg, hid_t h5_type, std::string h5_path, std::string in_bin_path)
        : config(cfg), h5_native_type(h5_type), h5_output_file(h5_path), input_bin_file(in_bin_path) 
    {
        dump_input_file = h5_output_file + ".input.bin";
        dump_recon_file = h5_output_file + ".recom.bin";
    }

    void run() {
        prepare_data();
        compress_write();
        decompress_read();
        verify();
    }

private:
    mans::h5::MansConfig config;
    hid_t h5_native_type;
    std::string h5_output_file;
    std::string input_bin_file;     
    std::string dump_input_file;    
    std::string dump_recon_file;    

    std::vector<T> host_data;
    std::vector<T> recon_data;
    double raw_size_mb = 0;

    // 1. Prepare Data
    void prepare_data() {
        std::cout << BOLD << "[1] Preparing Data..." << RST << "\n";
        if (!input_bin_file.empty()) {
            std::cout << "  -> Loading external file: " << input_bin_file << "\n";
            try {
                host_data = load_bin<T>(input_bin_file);
            } catch (std::exception& e) {
                std::cerr << YLW << "  -> [WARN] Failed to load file. Fallback to generator." << RST << "\n";
                input_bin_file = "";
            }
        }

        if (input_bin_file.empty()) {
            size_t num = 512 * 2000; // ~1 million elements
            std::cout << "  -> Generating synthetic data (" << num << " elements)...\n";
            host_data = generate_synthetic_data<T>(num, config.get_params().adm_threshold);
        }

        raw_size_mb = (host_data.size() * sizeof(T)) / 1048576.0;
        std::cout << "  -> Raw Size: " << BLU << raw_size_mb << " MB" << RST << "\n";
        
        save_bin(dump_input_file, host_data);
    }

    // 2. Compress & Write
    void compress_write() {
        std::cout << BOLD << "[2] Writing HDF5 (Compression)..." << RST << "\n";
        
        hid_t f_id = H5Fcreate(h5_output_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        CHECK_H5(f_id);

        hsize_t dims[1] = {host_data.size()};
        hid_t s_id = H5Screate_simple(1, dims, NULL);
        hid_t p_id = H5Pcreate(H5P_DATASET_CREATE);

        hsize_t chunk[1] = {512 * 64}; 
        CHECK_H5(H5Pset_chunk(p_id, 1, chunk));

        std::vector<unsigned int> cd = config.to_cd_values();
        CHECK_H5(H5Pset_filter(p_id, H5Z_FILTER_MANS_ID, 0, cd.size(), cd.data()));

        hid_t d_id = H5Dcreate2(f_id, "dataset", h5_native_type, s_id, H5P_DEFAULT, p_id, H5P_DEFAULT);

        auto t1 = std::chrono::high_resolution_clock::now();
        CHECK_H5(H5Dwrite(d_id, h5_native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, host_data.data()));
        CHECK_H5(H5Fflush(f_id, H5F_SCOPE_GLOBAL));
        auto t2 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double throughput = raw_size_mb / (ms/1000.0);
        
        std::cout << "  -> Write Time: " << BLU << ms << " ms" << RST 
                  << " (" << BLU << throughput << " MB/s" << RST << ")\n";

        H5Dclose(d_id); H5Pclose(p_id); H5Sclose(s_id); H5Fclose(f_id);

        size_t c_size = get_file_size(h5_output_file);
        double ratio = (double)(host_data.size()*sizeof(T))/c_size;

        std::cout << "  -> File Size:  " << BLU << c_size / 1024.0 << " KB" << RST 
                  << " (Ratio: " << BLU << ratio << "x" << RST << ")\n";
    }

    // 3. Decompress & Read
    void decompress_read() {
        std::cout << BOLD << "[3] Reading HDF5 (Decompression)..." << RST << "\n";
        recon_data.resize(host_data.size());

        hid_t f_id = H5Fopen(h5_output_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        CHECK_H5(f_id);
        hid_t d_id = H5Dopen2(f_id, "dataset", H5P_DEFAULT);

        auto t1 = std::chrono::high_resolution_clock::now();
        CHECK_H5(H5Dread(d_id, h5_native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, recon_data.data()));
        auto t2 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double throughput = raw_size_mb / (ms/1000.0);

        std::cout << "  -> Read Time:  " << BLU << ms << " ms" << RST 
                  << " (" << BLU << throughput << " MB/s" << RST << ")\n";

        H5Dclose(d_id); H5Fclose(f_id);

        save_bin(dump_recon_file, recon_data);
    }

    // 4. Verify
    void verify() {
        std::cout << BOLD << "[4] Verifying..." << RST << "\n";
        if (host_data == recon_data) {
            std::cout << "\n" << GRN << "  ****************************************" << RST << "\n";
            std::cout << GRN << "   [SUCCESS] Data Exact Match (Bit-Exact) " << RST << "\n";
            std::cout << GRN << "  ****************************************" << RST << "\n";
        } else {
            std::cerr << "\n" << RED << "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << RST << "\n";
            std::cerr << RED << "   [FAILURE] Data Mismatch Detected!      " << RST << "\n";
            std::cerr << RED << "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << RST << "\n";
            for(size_t i=0; i<host_data.size(); ++i) {
                if(host_data[i] != recon_data[i]) {
                    std::cerr << "  Mismatch at [" << i << "]: Orig=" 
                              << host_data[i] << " != Recon=" << recon_data[i] << "\n";
                    break; 
                }
            }
            exit(1);
        }
    }
};

// ==========================================
// Main Function
// ==========================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << YLW << "Usage: " << argv[0] << " <config_file> <output.h5> [input_file.bin]" << RST << "\n";
        return 1;
    }

    std::string config_file = argv[1];
    std::string h5_file = argv[2];
    std::string in_bin = (argc >= 4) ? argv[3] : "";

    mans::h5::MansConfig config;
    try { config.load(config_file); } 
    catch (std::exception& e) { std::cerr << RED << "Config Error: " << e.what() << RST << "\n"; return 1; }

    auto dtype = config.get_params().dtype;
    
    std::cout << BOLD << "========================================\n";
    std::cout << "    H5Z-MANS Integration Test Runner    \n";
    std::cout << "========================================" << RST << "\n";
    std::cout << " Config: " << config_file << "\n";
    std::cout << " Input:  " << (in_bin.empty() ? "(Auto-Gen)" : in_bin) << "\n";
    std::cout << " Type:   " << (dtype == mans::DataType::U16 ? "U16" : "U32") << "\n\n";

    if (dtype == mans::DataType::U16) {
        Tester<uint16_t> t(config, H5T_NATIVE_UINT16, h5_file, in_bin);
        t.run();
    } else {
        Tester<uint32_t> t(config, H5T_NATIVE_UINT32, h5_file, in_bin);
        t.run();
    }

    H5close();
    return 0;
}