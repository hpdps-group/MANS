// execute
// ./mapping input_file.quant_u2 output_file.quant_u2 uint16

#include <iostream>
#include <fstream>
#include <stdint.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>


static const int cmp_tblock_size = 32; // 32 threads
// static const int dec_tblock_size = 32; // 32 should be the best, not need to modify.
static const int cmp_chunk = 16;
static const int max_bytes_signal_per_ele_16b = 2;
// static const int cmpbytes_per_element_32b = 8;
// static const int dec_chunk = 1024;
// static const int aligned = 8;
static const int max_signals_16b = 3;
static const int max_signals_32b = 7;  
static const int decmp_chunk = 32;  // one decomp thread process decmp_chunk cmp thread
static const int warp_size = 32;
static const int k = 4;
static const int kn = 16;

__global__ void restore_signal(int* d_output_lengths, uint16_t* d_signal_length, uint8_t* d_concatenated_signals, uint8_t* d_signals, int data_size)
{

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    int lane = idx & 0x1f;
    int warp = idx >> 5;

    if(idx * cmp_chunk > data_size) return;

    int length = d_signal_length[warp];
    int src_start_idx = d_output_lengths[warp] * cmp_tblock_size + lane * length;
    int dst_start_idx = idx * cmp_chunk;

    uint8_t bit_buffer = 0;
    int signal_idx = -1;
    int offset_byte = 0;
    bool bit = 0;

    uint8_t local_signal[cmp_chunk] = {0};

    #pragma unroll
    for(; offset_byte < length && signal_idx < cmp_chunk; offset_byte++)
    {
        bit_buffer = d_concatenated_signals[src_start_idx + offset_byte];
        #pragma unroll
        for(int i = 7; i >= 0 && signal_idx < cmp_chunk; i--)
        {
            bit = (bit_buffer >> i) & 1;
            if (bit) // 检查最高位是否为1
            { 
                // 遇到新的信号起点，切换到下一个信号
                signal_idx ++;
            } else {
                local_signal[signal_idx]++;
            }
        }
    }

    int4* signal_int2 = reinterpret_cast<int4*>(d_signals + dst_start_idx);
    int4* local_signal_int2 = reinterpret_cast<int4*>(local_signal);

    #pragma unroll
    for (int i = 0; i < cmp_chunk / 16; ++i) {
        signal_int2[i] = local_signal_int2[i];
    }
}

__global__ void decompress_kernel_16b(uint16_t* decmp_data, uint16_t* centers, uint8_t* codes, uint8_t* signals, int shift) {
    const int block_id = blockIdx.x * blockDim.x + threadIdx.x;  // 每个线程对应一个 block
    const int lane = block_id & 0x1f;
    const int bid = blockIdx.x;

    int decompressed_idx = block_id * decmp_chunk;

    uint8_t local_codes[decmp_chunk];
    uint8_t local_signals[decmp_chunk];

    int4* local_codes_int2 = reinterpret_cast<int4*>(local_codes);
    int4* codes_int2 = reinterpret_cast<int4*>(codes + decompressed_idx);
    int4* local_signals_int2 = reinterpret_cast<int4*>(local_signals);
    int4* signals_int2 = reinterpret_cast<int4*>(signals + decompressed_idx);

    #pragma unroll
    for (int i = 0; i < decmp_chunk / 16; ++i) {
        local_codes_int2[i] = codes_int2[i];
        local_signals_int2[i] = signals_int2[i];
    } 

    uint16_t local_result[decmp_chunk];

    int  center = lane < 16 ? centers[bid * 2] : centers[bid * 2 + 1];

    uint8_t code = 0;
    uint8_t signal = 0;
    for(int i = 0; i < decmp_chunk; i++)
    {
        code = local_codes[i];
        signal = local_signals[i];

        int diff = (code % 2 == 1) ? ((code - 1) / 2) : ((code) / 2);
        diff += signal * 126;
        local_result[i] = (code % 2 == 1) ? center - diff : center + diff;
    }

    int4* local_result_int4 = reinterpret_cast<int4*>(local_result);
    int4* decmp_int4 = reinterpret_cast<int4*>(decmp_data + decompressed_idx);

    #pragma unroll
    for (int i = 0; i < decmp_chunk / 8; ++i) {
        decmp_int4[i] = local_result_int4[i];
    }  
}

__global__ void map_values_kernel_16b(
    const uint16_t* data, 
    uint8_t* code, 
    uint8_t* bit_signal, 
    uint16_t* centers, 
    uint16_t* signal_length ,
    volatile int* const __restrict__ locOffset, 
    volatile int* const __restrict__ cmpOffset, 
    volatile int* const __restrict__ flag, 
    int data_size, 
    int shift) 
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = cmp_chunk >> k;   

    int base_block_start_idx, base_block_end_idx;
    int base_thread_start_idx, base_thread_end_idx;
    uint4 tmp_buffer;
    
    uint8_t local_code[cmp_chunk] = {0};
    // uint8_t local_signal[cmp_chunk] = {0};
    uint8_t local_bit_signal[cmp_chunk * max_bytes_signal_per_ele_16b] = {0};

    int diff = 0;
    int output_idx = 0;
    int local_idx = 0;
    int bit_offset = 0; // current bit position
    uint16_t currValue = 0;

    base_block_start_idx = warp * cmp_chunk * cmp_tblock_size;
    base_block_end_idx = base_block_start_idx + cmp_chunk * cmp_tblock_size;
    base_thread_start_idx = base_block_start_idx + lane * cmp_chunk;
    base_thread_end_idx = base_thread_start_idx + cmp_chunk;

     // 每线程的局部和与计数
    uint32_t local_sum = 0;
    uint32_t local_count = 0;
 
     // 每个线程遍历部分数据，计算局部和与计数
     int end = min(base_thread_end_idx, data_size);
     for (int i = base_thread_start_idx; i < end; i++) {
        local_sum += __ldg(&data[i]);  // 使用 __ldg() 让全局内存加载更高效
        local_count++;
    }
 
     // 利用 Warp Shuffle 汇总所有线程的和与计数
    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }

 
     // Warp 内线程 0 计算平均值，并写入 center 数组
    int center = 0;
    center = local_count > 0 ? local_sum / local_count : 0;
    center = __shfl_sync(0xffffffff, center, 0);

    // if(warp ==1) printf("%d\n", center);
 

    bool is_center;
    uint16_t remain = 0;
    uint8_t res = 0;

    #pragma unroll
    for(int j = 0; j < block_num; j++)
    {
        base_thread_start_idx = base_block_start_idx + lane * cmp_chunk + j * kn;
        base_thread_end_idx = base_thread_start_idx + kn;
        end = min(base_thread_end_idx, data_size);
        if(base_thread_start_idx > data_size) break;

        #pragma unroll
        for(int i = base_thread_start_idx; i < end; i+=8)
        {
            tmp_buffer = reinterpret_cast<const uint4*>(data)[i / 8];

            currValue = static_cast<uint16_t>(tmp_buffer.x & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            
            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            // next
            local_idx++;
            bit_offset += output_idx;
            
            currValue = static_cast<uint16_t>((tmp_buffer.x >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>(tmp_buffer.y & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>((tmp_buffer.y >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>(tmp_buffer.z & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>((tmp_buffer.z >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>(tmp_buffer.w & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>((tmp_buffer.w >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            // next
            local_idx++;
            bit_offset += output_idx;
        }
    }


    // get max signal length in a warp, assume bitstream_length is the length of bit_signal(bytes)
    #pragma unroll 
    uint16_t max_bitstream_length = bit_offset;  // 初始化为当前线程的长度
    for (int offset = 16; offset > 0; offset /= 2) {
        max_bitstream_length = max(max_bitstream_length, __shfl_down_sync(0xFFFFFFFF, max_bitstream_length, offset));
    }
    max_bitstream_length = (max_bitstream_length + 7) / 8;  //bytes
    max_bitstream_length = __shfl_sync(0xffffffff, max_bitstream_length, 0);

    // save max length to signal_length
    // if (lane == 31) {
    //     signal_length[bid] = max_bitstream_length;
    //     centers[bid] = center; 
    // }


    int total_bits = max_bitstream_length * 8;
    int current_byte = bit_offset / 8;
    uint8_t mask = (0xFF >> (bit_offset % 8));
    (bit_offset < total_bits) ? 
        local_bit_signal[current_byte] = (bit_offset % 8 == 0) ? 0xFF : (local_bit_signal[current_byte] | mask) 
        : 0;

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        signal_length[bid] = max_bitstream_length;
        centers[bid] = center; 
        locOffset[warp+1] = max_bitstream_length;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            // Decoupled look-back
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                // Local sum not end.
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                // Lookback end.
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                // Continues lookback.
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        // Update global flag.
        if(!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if(!lane) flag[warp] = 2;
        __threadfence();  
    }
    __syncthreads();

    // Assigning compression bytes by given prefix-sum results.
    if(!lane) base_idx = excl_sum;
    __syncthreads();

    int write_pos = excl_sum * cmp_tblock_size + lane * max_bitstream_length;
    uint8_t* bit_ptr = bit_signal + write_pos;

    #pragma unroll
    for (int i = 0; i < max_bitstream_length; i++) {
        bit_signal[write_pos + i] = local_bit_signal[i];
    }

    // d_code
    int4* code_int2 = reinterpret_cast<int4*>(code + base_block_start_idx + lane * cmp_chunk);
    int4* local_code_int2 = reinterpret_cast<int4*>(local_code);

    #pragma unroll
    for (int i = 0; i < cmp_chunk / 16; ++i) {
        code_int2[i] = local_code_int2[i];
    }

}

__global__ void prefixsum(
    uint16_t* signal_length ,
    volatile int* const __restrict__ locOffset, 
    volatile int* const __restrict__ cmpOffset, 
    volatile int* const __restrict__ flag
)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;

    int max_bitstream_length = signal_length[warp];

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        locOffset[warp+1] = max_bitstream_length;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            // Decoupled look-back
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                // Local sum not end.
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                // Lookback end.
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                // Continues lookback.
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        // Update global flag.
        if(!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if(!lane) flag[warp] = 2;
        __threadfence();  
    }
    __syncthreads();

}
// Template helper function to load binary file into memory based on data type
template <typename T>
T* load_file(const char* filename, size_t& size) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }

    size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = size / sizeof(T);
    T* data = new T[num_elements];
    file.read(reinterpret_cast<char*>(data), size);
    file.close();
    
    return data;
}



// Helper function to save result to binary file
void save_file(const char* filename, const uint8_t* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), size);
    file.close();
}


__global__ void concat(
    const uint8_t* d_bit_signals,  // 原始 bitstream 数据
    const uint16_t* d_signal_length,   // 每个 warp 的 bitstream 长度
    const int* d_output_lengths,  // 前缀和数组，存放目标偏移
    uint8_t* d_concatenated_output,  // 目标拼接后数据
    int gsize,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算索引
    int lane = idx & 0x1f;
    int warp = idx >> 5;

    if(idx * cmp_chunk < num_elements)
    {
        int length = d_signal_length[warp];
        int bit_signal_start = idx * cmp_chunk * max_bytes_signal_per_ele_16b;
        int concat_start = d_output_lengths[warp] * cmp_tblock_size + lane * length;

        for(int i = 0; i < length; i++)
        {
            d_concatenated_output[concat_start + i] = d_bit_signals[bit_signal_start + i];
        }     
    }
}

__global__ void convert_uint16_to_int(const uint16_t* input, int* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = static_cast<int>(input[idx]);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input file> <output file> <data type>" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    std::string data_type = argv[3]; // 从命令行获取数据类型

    int shift = 1;

    std::cout << "Input file: " << input_file << std::endl;

    // Load input file into host memory based on data type
    size_t data_size = 0;
    void* h_data = nullptr;
    int num_elements = 0;


    if (data_type == "uint16") {
        h_data = load_file<uint16_t>(input_file, data_size);
        num_elements = data_size / sizeof(uint16_t);
    } else if (data_type == "uint32") {
        h_data = load_file<uint32_t>(input_file, data_size);
        num_elements = data_size / sizeof(uint32_t);
    }

    if (!h_data) {
        std::cerr << "Error: Failed to load data from file: " << input_file << std::endl;
        return 1;
    }

    std::cout << "elements num: " << num_elements << std::endl;

    // 配置CUDA内核
    int bsize = cmp_tblock_size;
    int gsize = (num_elements + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);

    printf("gsize:%d\n", gsize);

    // printf("block: %d threads: %d\n", gsize, bsize * gsize);

    // define 
    void* d_data;
    void* d_decmpdata;
    void* d_cmpdata;

    // compressed struct: header(signal_length & center) & code & bit_signal
    // int* d_offsets
    uint16_t* d_signal_length;
    void* d_centers;
    uint8_t* d_codes;
    uint8_t* d_bit_signals; 
    uint8_t* d_concatenated_signals;
    int* d_locOffset;
    int* d_flag;

    // memory allocation
    cudaMalloc((void**)&d_data, data_size);
    cudaMalloc((void**)&d_signal_length, gsize * sizeof(uint16_t));
    cudaMemset(d_signal_length, 0, gsize * sizeof(uint16_t));
    cudaMalloc((void**)&d_codes, bsize * gsize * cmp_chunk * sizeof(uint8_t));
    cudaMalloc((void**)&d_locOffset, (gsize + 1) * sizeof(int));
    cudaMalloc((void**)&d_flag, (gsize + 1) * sizeof(int));
    if (data_type == "uint16") {
        // cudaMalloc(&d_cmpdata, bsize * gsize * cmp_chunk * sizeof(uint16_t));
        cudaMalloc(&d_decmpdata, bsize * gsize * cmp_chunk * sizeof(uint16_t));

        cudaMalloc(&d_centers, gsize * sizeof(uint16_t));  
        cudaMalloc(&d_bit_signals, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));
        cudaMemset(d_bit_signals, 0, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));
        cudaMalloc(&d_concatenated_signals, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));
        cudaMemset(d_concatenated_signals, 255, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));
    } else if (data_type == "uint32") {
        // cudaMalloc(&d_decmpdata, bsize * gsize * cmp_chunk * sizeof(uint32_t));

        // cudaMalloc(&d_centers, gsize * sizeof(uint32_t));
        // cudaMalloc(&d_bit_signals, bsize * gsize * cmp_chunk * sizeof(uint8_t));
    }

    // mem copy
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

    //prefix-sum initialize
    int* d_output_lengths;
    cudaMalloc(&d_output_lengths, (gsize + 1) * sizeof(int));
    int* d_signal_length_int;
    cudaMalloc(&d_signal_length_int, gsize * sizeof(int));
    thrust::device_ptr<int> dev_output_lengths(d_output_lengths);
    thrust::device_ptr<int> dev_signal_length(d_signal_length_int);

    // warmup
    for(int i = 0; i < 0; i++)
    {
        if (data_type == "uint16") {
            map_values_kernel_16b<<<gsize, bsize>>>(
                (uint16_t*)d_data, d_codes, d_bit_signals, (uint16_t*)d_centers, d_signal_length, d_locOffset, d_output_lengths, d_flag, num_elements, shift);
                cudaMemset(d_flag, 0, (gsize + 1) * sizeof(int));
        } else if (data_type == "uint32") {
            // map_values_kernel_32b<<<gsize, bsize>>>(
            //     (uint32_t*)d_data, d_result, nullptr, d_output_lengths, d_center, num_elements, shift, signal);
        }
    }

    // 设置CUDA事件用于测量时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // adm time
    cudaEventRecord(start);

    if (data_type == "uint16") {
        map_values_kernel_16b<<<gsize, bsize, sizeof(unsigned int)*2, stream>>>(
            (uint16_t*)d_data, d_codes, d_concatenated_signals, (uint16_t*)d_centers, d_signal_length, d_locOffset, d_output_lengths, d_flag, num_elements, shift);
    } else if (data_type == "uint32") {
        // map_values_kernel_32b<<<gsize, bsize>>>(
        //     (uint32_t*)d_data, d_result, nullptr, d_output_lengths, d_center, num_elements, shift, signal);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float adm_milliseconds = 0;
    cudaEventElapsedTime(&adm_milliseconds, start, stop);

    // print_device_arrays(d_signal_length, d_output_lengths, 3, 3);

    // int threads = 256;
    // int blocks = (gsize + threads - 1) / threads;
    // convert_uint16_to_int<<<blocks, threads>>>(d_signal_length, d_signal_length_int, gsize);
    // cudaDeviceSynchronize();

    // prefix-sum time
    cudaMemset(d_flag, 0, (gsize + 1) * sizeof(int));
    cudaEventRecord(start);

    prefixsum<<<gsize, bsize>>>(d_signal_length, d_locOffset, d_output_lengths, d_flag);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float prefixsum_milliseconds = 0;
    cudaEventElapsedTime(&prefixsum_milliseconds, start, stop);

    // // print_device_arrays(d_signal_length, d_output_lengths, 3, 3);

    // // int* h_output_lengths = new int[gsize];
    // // cudaMemcpy(h_output_lengths, d_output_lengths, gsize * sizeof(int), cudaMemcpyDeviceToHost);
    // // for(int i = 40070; i < 40093; i++) {
    // //     printf("d_output_lengths[%d] = %d\n", i, h_output_lengths[i]);
    // // }
    // // delete[] h_output_lengths;


    // //concat time
    // cudaEventRecord(start);
    // concat<<<gsize, bsize>>>(
    //     d_bit_signals, d_signal_length, d_output_lengths, d_concatenated_signals, gsize, num_elements);
    // cudaDeviceSynchronize();

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float concat_milliseconds = 0;
    // cudaEventElapsedTime(&concat_milliseconds, start, stop);

    // print_device_arrays(d_signal_length, d_output_lengths, 3, 3);

    // 复制局部长度和sum长度到主机
    uint16_t* h_signal_length = new uint16_t[gsize];
    int* h_output_lengths = new int[gsize];
    cudaMemcpy(h_signal_length, d_signal_length, gsize * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_lengths, d_output_lengths, gsize * sizeof(int), cudaMemcpyDeviceToHost);


    // 计算总的输出大小
    size_t bit_signals_size = (h_signal_length[gsize - 1] + h_output_lengths[gsize - 1]) * cmp_tblock_size * sizeof(uint8_t);

    //merger into d_cmpdata
    size_t signal_length_size = gsize * sizeof(uint16_t);  // d_signal_length
    size_t centers_size = 0;
    if (data_type == "uint16") {
        centers_size = gsize * sizeof(uint16_t);
    } else if (data_type == "uint32") {
        centers_size = gsize * sizeof(uint32_t);
    }
    size_t codes_size = num_elements * sizeof(uint8_t);           // d_codes

    // 计算总大小
    size_t total_size = signal_length_size + centers_size + codes_size + bit_signals_size;

    // 在设备上分配统一数组
    cudaMalloc(&d_cmpdata, total_size);

    // 计算子数组的指针位置
    uint16_t* d_signal_length_ptr = reinterpret_cast<uint16_t*>(d_cmpdata);
    void* d_centers_ptr = reinterpret_cast<uint8_t*>(d_cmpdata) + signal_length_size;
    uint8_t* d_codes_ptr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(d_cmpdata) + signal_length_size + centers_size);
    uint8_t* d_bit_signals_ptr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(d_cmpdata) + signal_length_size + centers_size + codes_size);

    // **将已有数据拷贝到 d_cmpdata**
    cudaMemcpy(d_signal_length_ptr, d_signal_length, signal_length_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_centers_ptr, d_centers, centers_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_codes_ptr, d_codes, codes_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_bit_signals_ptr, d_concatenated_signals, bit_signals_size, cudaMemcpyDeviceToDevice);

    uint8_t* h_cmpdata = new uint8_t[total_size];

    cudaMemcpy(static_cast<uint8_t*>(h_cmpdata), d_cmpdata, total_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // save
    save_file(output_file, h_cmpdata, total_size);

    // 计算内核吞吐量 (数据量 / 时间)
    printf("\033[0;34m-------Cmp Throughput-------\033[0m\n") ;
    float adm_throughput = (data_size / 1024.0 / 1024 / 1024) / (adm_milliseconds / 1000.0f);  // 单位：字节每秒
    float prefixsum_throughput = (data_size / 1024.0 / 1024 / 1024) / (prefixsum_milliseconds / 1000.0f); 
    // float concat_throughput = (data_size / 1024.0 / 1024 / 1024) / (concat_milliseconds / 1000.0f); 
    // float total_cmp_throughput = (data_size / 1024.0 / 1024 / 1024) / ((adm_milliseconds + prefixsum_milliseconds + concat_milliseconds) / 1000.0f); 
    // float total_cmp_throughput = (data_size / 1024.0 / 1024 / 1024) / ((adm_milliseconds) / 1000.0f); 
    // printf("Total Cmp throughput: %.2f GB/s\n", total_cmp_throughput);
    printf("ADM Kernel throughput: %.2f GB/s\n", adm_throughput);
    printf("Prefix-sum Kernel throughput: %.2f GB/s\n", prefixsum_throughput);
    // printf("Concat Kernel throughput: %.2f GB/s\n", concat_throughput);

    printf("\033[0;34m-------Cmp Time-------\033[0m\n") ;
    // printf("Total Cmp Time: %.2f ms\n", (adm_milliseconds + prefixsum_milliseconds + concat_milliseconds));
    printf("ADM cost %.2f ms\n",adm_milliseconds);
    printf("Prefix-sum cost %.2f ms\n",prefixsum_milliseconds);
    // printf("Concat cost %.2f ms\n",concat_milliseconds);

    // ----------------Decmp------------------
    // d_cmpdata = d_signal_length + d_centers + d_codes + d_concatenated_signals

    // decmp kernel 1: prefix-sum
    // prefixsum_milliseconds = prefixsum_milliseconds;

    // decmp kernel 2: restore signal
    // threads = 256;
    // blocks = (gsize + threads - 1) / threads;
    uint8_t* d_signals;
    cudaMalloc(&d_signals, bsize * gsize * cmp_chunk * sizeof(uint8_t));
    cudaEventRecord(start);
    restore_signal<<<gsize, bsize>>>(d_output_lengths, d_signal_length, d_concatenated_signals, d_signals, num_elements);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float restore_milliseconds = 0;
    cudaEventElapsedTime(&restore_milliseconds, start, stop);

    // decmp kernel 3: decmp adm
    int decmp_gsize = (num_elements + bsize * decmp_chunk - 1) / (bsize * decmp_chunk);

    // printf("%d %d\n", gsize, decmp_gsize);

    cudaEventRecord(start);
    if (data_type == "uint16") {
        decompress_kernel_16b<<<decmp_gsize, bsize>>>((uint16_t*)d_decmpdata, (uint16_t*)d_centers, d_codes, d_signals, shift);
    } else if (data_type == "uint32") {
        // decompress_kernel_32b<<<decmp_gsize, bsize>>>(d_output, d_offsets, d_output_lengths, (uint32_t*)d_decmpdata, d_center, shift, signal);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float decmp_adm_milliseconds = 0;
    cudaEventElapsedTime(&decmp_adm_milliseconds, start, stop);

    printf("\033[0;34m-------Decmp Throughput-------\033[0m\n") ;
    float restore_throughput = (data_size / 1024.0 / 1024 / 1024) / (restore_milliseconds / 1000.0f);  // 单位：字节每秒
    float decmp_adm_throughput = (data_size / 1024.0 / 1024 / 1024) / (decmp_adm_milliseconds / 1000.0f); 
    // float total_decmp_throughput = (data_size / 1024.0 / 1024 / 1024) / ((restore_milliseconds + prefixsum_milliseconds + decmp_adm_milliseconds) / 1000.0f); 
    // printf("Total Decmp throughput: %.2f GB/s\n", total_decmp_throughput);
    // printf("Prefix-sum Kernel throughput: %.2f GB/s\n", prefixsum_throughput);
    printf("Restore Signal Kernel throughput: %.2f GB/s\n", restore_throughput);
    printf("Decmp ADM throughput: %.2f GB/s\n", decmp_adm_throughput);

    printf("\033[0;34m-------Decmp Time-------\033[0m\n") ;
    // printf("Total cost %.2f ms\n",(restore_milliseconds + prefixsum_milliseconds + decmp_adm_milliseconds));
    // printf("Prefix-sum cost %.2f ms\n",prefixsum_milliseconds);
    printf("Restore cost %.2f ms\n",restore_milliseconds);
    printf("Decmp ADM cost %.2f ms\n",decmp_adm_milliseconds);

    printf("\033[0;34m-------Data Size-------\033[0m\n");
    printf("Original size: %d Bytes\n",data_size);
    printf("Output size: %d Bytes\n",total_size);
    printf("CR : %.2f x\n",data_size * 1.0 / total_size);
    printf("Signal Length size: %d Bytes\n",signal_length_size);
    printf("Centers size: %d Bytes\n",centers_size);
    printf("Codes size: %d Bytes\n",codes_size);
    printf("Bit-signal size: %d Bytes\n",bit_signals_size);

    //verify lossless
    void* h_decmpdata = nullptr;

    if (data_type == "uint16") {
        h_decmpdata = new uint16_t[num_elements];
        cudaMemcpy(static_cast<uint16_t*>(h_decmpdata), d_decmpdata, num_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    } else if (data_type == "uint32") {
        h_decmpdata = new uint32_t[num_elements];
        cudaMemcpy(static_cast<uint32_t*>(h_decmpdata), d_decmpdata, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    bool test = true;
    if (data_type == "uint16") {
        uint16_t* decompressed_data = static_cast<uint16_t*>(h_decmpdata); // 提前转换类型
        uint16_t* original_data = static_cast<uint16_t*>(h_data);         // 如果 h_data 也是 void*
        for (int i = 0; i < num_elements; i++) {
            if (decompressed_data[i] != original_data[i]) {
                printf("\033[0;31mFail error check!\033[0m\n");
                printf("\033[0;31mError: Data mismatch at index %d, original = %d, decompressed = %d\033[0m\n", 
                       i, original_data[i], decompressed_data[i]);
                test = false;
                break;
            }
        }
    } else if (data_type == "uint32") {
        uint32_t* decompressed_data = static_cast<uint32_t*>(h_decmpdata); // 提前转换类型
        uint32_t* original_data = static_cast<uint32_t*>(h_data);         // 如果 h_data 也是 void*
        for (int i = 0; i < num_elements; i++) {
            if (decompressed_data[i] != original_data[i]) {
                printf("\033[0;31mFail error check!\033[0m\n");
                printf("\033[0;31mError: Data mismatch at index %d, original = %d, decompressed = %d\033[0m\n", 
                       i, original_data[i], decompressed_data[i]);
                test = false;
                break;
            }
        }
    }

    if(test) printf("\033[0;32mPass error check!\033[0m\n");

    // // 设定打印范围
    // int debug_start_idx = 512;   // 可以修改为任意起始索引
    // int debug_end_idx = 515;     // 需要检查的数据长度

    // // 申请 host 端数组
    // uint16_t* h_centers = new uint16_t[gsize];
    // uint8_t* h_codes = new uint8_t[num_elements];
    // uint8_t* h_signals = new uint8_t[num_elements];

    // // 将数据从 GPU 复制到 CPU
    // cudaMemcpy(h_centers, d_centers, gsize * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_codes, d_codes, num_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_signals, d_signals, num_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);


    // // 打印 Centers
    // uint16_t* original_data = static_cast<uint16_t*>(h_data);

    // printf("%d\n", h_centers[0]);
    // printf("%d\n", h_centers[1]);
    // printf("==== datas (部分数据) ====\n");
    // for (int i = debug_start_idx; i < debug_end_idx; i++) {  // 仅打印最多16个中心值
    //     printf("[%d]: %d\n", i, original_data[i]);
    // }
    // printf("\n");

    // // 打印 Codes
    // printf("==== d_codes (部分数据) ====\n");
    // for (int i = debug_start_idx; i < debug_end_idx; i++) {
    //     printf("[%d]: %u\n", i, h_codes[i]);
    // }
    // printf("\n");

    // // 打印 Signals
    // printf("==== d_signals (部分数据) ====\n");
    // for (int i = debug_start_idx; i < debug_end_idx; i++) {
    //     printf("[%d]: %u\n", i, h_signals[i]);
    // }
    // printf("\n");

    // // 释放 CPU 端内存
    // delete[] h_centers;
    // delete[] h_codes;
    // delete[] h_signals;

    // 清理内存
    cudaFree(d_data);
    cudaFree(d_decmpdata);
    cudaFree(d_cmpdata);
    cudaFree(d_signal_length);
    cudaFree(d_centers);
    cudaFree(d_codes);
    cudaFree(d_bit_signals);
    cudaFree(d_concatenated_signals);
    cudaFree(d_output_lengths);

    delete[] static_cast<uint8_t*>(h_cmpdata);
    delete[] static_cast<uint16_t*>(h_data);
    delete[] h_signal_length;
    delete[] h_output_lengths;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    printf("Memory has been successfully freed.\n");

    return 0;
}