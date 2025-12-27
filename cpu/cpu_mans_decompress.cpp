// compiler: g++ -std=c++17 -O3 cpu_mans_decompress.cpp mans_cpu.cpp -o cpu_mans_decompress -fopenmp
// exec    : OMP_NUM_THREADS=4 ./cpu_mans_decompress u2 input.bin output.u2 0
//           OMP_NUM_THREADS=4 ./cpu_mans_decompress u4 input.bin output.u4 1

#include <iostream>
#include <string>
#include <vector>


#include "../mans_defs.h"
#include "mans_cpu.h"
#include "file_utils.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Use: " << argv[0]
                  << " <u2|u4> <input_bin_file> <output_u2/u4_file> <save_adm>\n";
        return 1;
    }

    std::string dtype_str   = argv[1];
    std::string input_file  = argv[2];
    std::string output_file = argv[3];
    std::string save_flag   = argv[4];
    bool save_adm = (save_flag == "1");

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

    // 2. load data
    std::vector<uint8_t> input_data;
    if (!load_u8_file(input_file, input_data)) {
        std::cerr << "Failed to load input file: " << input_file << "\n";
        return 1;
    }
    if (input_data.empty()) {
        std::cerr << "Input file is empty.\n";
        return 1;
    }

    std::cout << "Decompressing to " << dtype_str << " (Input size: " << input_data.size() << ")...\n";

    std::vector<uint8_t> output_bytes;
    
    // Core decompress: set debug parameters (save_adm, dump_path, open_benchmark = true)
    mans::cpu::decompress_internal(
        input_data.data(),    // const void* input_data
        input_data.size(),    // size_t length
        params,
        output_bytes,
        save_adm,
        output_file + ".adm", // debug path: output.u2.adm
        true                  // open_benchmark = true
    );

    // 4. Save the result
    // Note: the internal interface has already converted vector<u16/u32> into vector<u8> (byte stream),
    // so we can write it directly using save_u8_file.
    if (!save_u8_file(output_file, output_bytes)) {
        std::cerr << "Failed to write output file: " << output_file << "\n";
        return 1;
    }

    std::cout << "Mans decompress finished! Output: " << output_file 
              << " (Size: " << output_bytes.size() << ")\n";

    return 0;
}