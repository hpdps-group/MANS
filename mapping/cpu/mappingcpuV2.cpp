// 编译： g++ -std=c++17 -mavx512f -fopenmp -march=native mappingcpuV2.cpp -o mappingcppV2
// 执行： ./mappingcppV1 ../../data/exafel/exafel_59200x388.u2 ../../data/exafel/testcpu.bin
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <chrono>

#include <immintrin.h>

// 全局参数
const int cmp_tblock_size = 32;
const int cmp_chunk = 16;
const int decmp_chunk = 32;
const int max_bytes_signal_per_ele_16b = 2;
const int warp_size = 32;

bool load_file(const std::string& filename, std::vector<uint16_t>& data) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;
    size_t size = in.tellg();
    in.seekg(0, std::ios::beg);
    data.resize(size / sizeof(uint16_t));
    in.read(reinterpret_cast<char*>(data.data()), size);
    return true;
}

void save_file(const std::string& filename, const std::vector<uint8_t>& data) {
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
}

void compress(
    const std::vector<uint16_t>& input_data,
    std::vector<int>& output_lengths,
    std::vector<uint16_t>& centers,
    std::vector<uint8_t>& codes,
    std::vector<uint8_t>& bit_signals
) {
    int num_elements = input_data.size();
    int gsize = (num_elements + cmp_tblock_size * cmp_chunk - 1) / (cmp_tblock_size * cmp_chunk);
    int total_threads = gsize * cmp_tblock_size;

    std::vector<int> signal_length(gsize, 0);
    std::vector<int> bit_offsets(total_threads, 0);
    centers.resize(gsize);
    codes.resize(num_elements);

    // Center calculation: parallelizing and reducing unnecessary work
    #pragma omp parallel for
    for (int warp = 0; warp < gsize; ++warp) {
        int base_idx = warp * cmp_tblock_size * cmp_chunk;
        int end_idx = std::min(base_idx + cmp_tblock_size * cmp_chunk, num_elements);

        uint64_t sum = 0;
        for (int i = base_idx; i < end_idx; ++i) {
            sum += input_data[i];
        }

        int count = end_idx - base_idx;
        centers[warp] = (count > 0) ? sum / count : 0;
    }

    // Allocate temporary buffer for bit_signals
    std::vector<uint8_t> tmp_bit_signals(total_threads * cmp_chunk * max_bytes_signal_per_ele_16b, 0);

    // Encoding and setting codes, bit_signals (in temporary space)
    #pragma omp parallel for
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int lane = thread_idx % cmp_tblock_size;
        int base_idx = warp * cmp_tblock_size * cmp_chunk + lane * cmp_chunk;

        if (base_idx >= num_elements) continue;
        int center = centers[warp];

        uint8_t* bit_out = &tmp_bit_signals[thread_idx * cmp_chunk * max_bytes_signal_per_ele_16b];

        int bit_offset = 0;

        for (int i = 0; i < cmp_chunk && base_idx + i < num_elements; ++i) {
            uint16_t val = input_data[base_idx + i];
            int diff = val > center ? val - center : center - val;
            int output_len = (val == center) ? 1 : (diff + 125) / 126;
            uint8_t res = (val == center) ? 1 : ((diff + 126 - output_len * 126) * 2 + (val > center ? -1 : 0) + 1);

            codes[base_idx + i] = res;

            // Set bitstream (mark the corresponding bit)
            bit_out[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            bit_offset += output_len;
        }

        bit_offsets[thread_idx] = bit_offset;
        int length_bytes = (bit_offset + 7) / 8;
        signal_length[warp] = std::max(signal_length[warp], length_bytes);
    }

    // Fill in the tail bits
    #pragma omp parallel for
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int bit_offset = bit_offsets[thread_idx];
        int max_len_bytes = signal_length[warp];

        if (bit_offset < max_len_bytes * 8) {
            uint8_t* bit_out = &tmp_bit_signals[thread_idx * cmp_chunk * max_bytes_signal_per_ele_16b];
            int byte_idx = bit_offset / 8;
            uint8_t mask = (bit_offset % 8 == 0) ? 0xFF : (0xFF >> (bit_offset % 8));
            bit_out[byte_idx] |= mask;
        }
    }

    // Compute prefix sum (serially)
    output_lengths.resize(gsize + 1);
    output_lengths[0] = 0;
    for (int i = 1; i <= gsize; ++i) {
        output_lengths[i] = output_lengths[i - 1] + signal_length[i - 1];
    }

    // Write back bit_signals
    int total_bit_bytes = output_lengths[gsize] * cmp_tblock_size;
    bit_signals.resize(total_bit_bytes, 0);

    #pragma omp parallel for
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int lane = thread_idx % cmp_tblock_size;
        int bit_len = signal_length[warp];
        int dst_base = output_lengths[warp] * cmp_tblock_size + lane * bit_len;

        if (dst_base + bit_len > total_bit_bytes) continue;

        const uint8_t* src = &tmp_bit_signals[thread_idx * cmp_chunk * max_bytes_signal_per_ele_16b];
        // 使用向量化指令进行批量拷贝
        #pragma omp simd
        for (int i = 0; i < bit_len; ++i) {
            bit_signals[dst_base + i] = src[i];
        }
    }
}


void decompress(
    const std::vector<int>& output_lengths,             // gsize
    const std::vector<uint16_t>& centers,               // gsize
    const std::vector<uint8_t>& codes,                  // num_elements
    const std::vector<uint8_t>& bit_signals,            // bitstream
    std::vector<uint16_t>& output_data                  // output: num_elements
)
{
    int num_elements = codes.size();
    int gsize = output_lengths.size();
    int total_threads = gsize * cmp_tblock_size;

    // Step 1: Restore signal[]
    std::vector<uint8_t> signals(num_elements, 0);

    #pragma omp parallel for
    for (int tid = 0; tid < total_threads; ++tid) {
        int warp = tid / cmp_tblock_size;
        int lane = tid % cmp_tblock_size;
        int idx = tid;

        if (idx * cmp_chunk >= num_elements) continue;

        int length = output_lengths[warp + 1] - output_lengths[warp];

        int src_start_idx = output_lengths[warp] * cmp_tblock_size + lane * length;
        int dst_start_idx = idx * cmp_chunk;

        uint8_t bit_buffer = 0;
        int signal_idx = -1;
        int offset_byte = 0;
        bool bit = 0;

        uint8_t local_signal[cmp_chunk] = {0};

        for (; offset_byte < length && signal_idx < cmp_chunk; offset_byte++) {
            bit_buffer = bit_signals[src_start_idx + offset_byte];
            for (int i = 7; i >= 0 && signal_idx < cmp_chunk; i--) {
                bit = (bit_buffer >> i) & 1;
                if (bit) {
                    signal_idx++;
                } else {
                    local_signal[signal_idx]++;
                }
            }
        }

        // Use a local copy to avoid accessing shared memory repeatedly
        for (int i = 0; i < cmp_chunk && dst_start_idx + i < num_elements; ++i) {
            signals[dst_start_idx + i] = local_signal[i];
        }
    }

    // Step 2: Decode values
    output_data.resize(num_elements);

    #pragma omp parallel for
    for (int tid = 0; tid < total_threads; ++tid) {
        int block_id = tid;
        int lane = block_id % warp_size;
        int bid = block_id / cmp_tblock_size;
        int base_idx = block_id * decmp_chunk;

        if (base_idx >= num_elements) continue;

        uint16_t center = (lane < 16) ? centers[bid * 2] : centers[bid * 2 + 1];

        // Use local variables to minimize memory access and reduce branch conditions
        for (int i = 0; i < decmp_chunk && base_idx + i < num_elements; ++i) {
            uint8_t code = codes[base_idx + i];
            uint8_t signal = signals[base_idx + i];

            int diff = (code % 2 == 1) ? ((code - 1) / 2) : (code / 2);
            diff += signal * 126;

            uint16_t val = (code % 2 == 1) ? center - diff : center + diff;
            output_data[base_idx + i] = val;
        }
    }
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Use: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    // 加载输入数据
    std::vector<uint16_t> input_data;
    if (!load_file(input_file, input_data)) {
        std::cerr << "Failed to load input file" << std::endl;
        return 1;
    }

    size_t num_elements = input_data.size();
    int gsize = (num_elements + cmp_tblock_size * cmp_chunk - 1) / (cmp_tblock_size * cmp_chunk);

    std::vector<int> output_lengths(gsize + 1);
    std::vector<uint16_t> centers(gsize);
    std::vector<uint8_t> codes(num_elements);
    std::vector<uint8_t> bit_signals;  // bitstream 输出会动态增长
    int last_length = 0;

    for(int i = 0; i < 5; i++)
    {
        compress(input_data, output_lengths, centers, codes, bit_signals);
    }

    // 压缩
    printf("\033[0;36m=======>Start ADM<=======\033[0m\n");
    int times = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < times; i++)
    {
        compress(input_data, output_lengths, centers, codes, bit_signals);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = (end - start) / times;
    printf("compress cost %.2f ms, throughput %.2f MB/s\n", duration.count(), num_elements * 2 / 1024 / 1024 / (duration.count() / 1000));

    // 合并为连续数据结构
    size_t len1 = output_lengths.size() * sizeof(int);
    size_t len2 = centers.size() * sizeof(uint16_t);
    size_t len3 = codes.size() * sizeof(uint8_t);
    size_t len4 = bit_signals.size();

    size_t total_size = len1 + len2 + len3 + len4;
    std::vector<uint8_t> merged(total_size);

    size_t offset = 0;
    std::memcpy(merged.data() + offset, output_lengths.data(), len1); offset += len1;
    std::memcpy(merged.data() + offset, centers.data(), len2);        offset += len2;
    std::memcpy(merged.data() + offset, codes.data(), len3);          offset += len3;
    std::memcpy(merged.data() + offset, bit_signals.data(), len4);

    save_file(output_file, merged);
    std::cout << "Compress finished! Write to " << output_file << std::endl;

    // 解压
    printf("\033[0;36m=======>Start DeADM<=======\033[0m\n");
    std::vector<uint16_t> recovered(num_elements);
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < times; i++)
    {
        decompress(output_lengths, centers, codes, bit_signals, recovered);
    }
    end = std::chrono::high_resolution_clock::now();    // 计时结束
    duration = (end - start) / times;
    printf("decompress cost %.2f ms, throughput %.2f MB/s\n", duration.count(), num_elements * 2 / 1024 / 1024 / (duration.count() / 1000));

    // 验证
    bool pass = true;
    for (size_t i = 0; i < num_elements; ++i) {
        if (input_data[i] != recovered[i]) {
            printf("\033[0;31mFail error check!\033[0m\n");
            printf("\033[0;31mError: Data mismatch at index %d, original = %d, decompressed = %d\033[0m\n", 
                i, input_data[i], recovered[i]);
            pass = false;
            break;
        }
    }

    if (pass) printf("\033[0;32mPass error check!\033[0m\n");

    return 0;
}
