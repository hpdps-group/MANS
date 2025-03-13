// execute
// ./mapping input_file.quant_u2 output_file.quant_u2 uint16 512

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
static const int cmpbytes_per_element_16b = 4;
static const int cmpbytes_per_element_32b = 8;
// static const int dec_chunk = 1024;
static const int aligned = 8;
static const int max_signals_16b = 3;
static const int max_signals_32b = 7;  
static const int decmp_chunk = 2;  // one decomp thread process decmp_chunk cmp thread
static const int warp_size = 32;


__global__ void map_values_kernel_16b(const uint16_t* data, uint8_t* code, uint8_t* signal, int* centers, int data_size, int shift) 
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = 1;   

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    uint4 tmp_buffer;
    
    uint8_t local_code[cmp_chunk] = {0};
    uint8_t local_signal[cmp_chunk] = {0};

    int diff = 0;
    int output_idx = 0;
    int local_idx = 0;
    uint16_t currValue = 0;

    base_start_idx = warp * cmp_chunk * 32;
    base_block_start_idx = base_start_idx + lane * cmp_chunk;
    base_block_end_idx = base_block_start_idx + cmp_chunk;

     // 每线程的局部和与计数
    uint32_t local_sum = 0;
    uint32_t local_count = 0;
 
     // 每个线程遍历部分数据，计算局部和与计数
    for (int i = base_block_start_idx; i < base_block_end_idx && i < data_size; i += 1) {
        local_sum += data[i]; // 累加数据值
        local_count++;        // 累加数据计数
    }
 
     // 利用 Warp Shuffle 汇总所有线程的和与计数
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
 
     // Warp 内线程 0 计算平均值，并写入 center 数组
    if (lane == 0) {
        if (local_count > 0) {
            centers[bid] = local_sum / local_count; // 存储平均值
        } else {
            centers[bid] = 0; // 避免除以 0
        }
    }

    if(idx == 0) printf("%d\n", centers[0]);
 
    __syncthreads();
 
     // 使用 `center[bid]` 继续后续计算
    int center = centers[bid];

    bool is_center;
    uint16_t remain = 0;
    uint8_t res = 0;

    for(int j = 0; j < block_num; j++)
    {
        if(base_block_start_idx > data_size) break;

        // #pragma unroll
        for(int i = base_block_start_idx; i < base_block_end_idx && i < data_size; i+=8)
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
            
            output_idx -= 1;
            local_code[local_idx] = is_center ? shift : res;
            local_signal[local_idx] = (uint8_t)((output_idx << 1));; 
            local_idx++;
            
            

            currValue = static_cast<uint16_t>((tmp_buffer.x >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            output_idx -= 1;
            local_code[local_idx] = is_center ? shift : res;
            local_signal[local_idx] = (uint8_t)((output_idx << 1));; 
            local_idx++;

            currValue = static_cast<uint16_t>(tmp_buffer.y & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            output_idx -= 1;
            local_code[local_idx] = is_center ? shift : res;
            local_signal[local_idx] = (uint8_t)((output_idx << 1));; 
            local_idx++;

            currValue = static_cast<uint16_t>((tmp_buffer.y >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            output_idx -= 1;
            local_code[local_idx] = is_center ? shift : res;
            local_signal[local_idx] = (uint8_t)((output_idx << 1));; 
            local_idx++;

            currValue = static_cast<uint16_t>(tmp_buffer.z & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            output_idx -= 1;
            local_code[local_idx] = is_center ? shift : res;
            local_signal[local_idx] = (uint8_t)((output_idx << 1));; 
            local_idx++;

            currValue = static_cast<uint16_t>((tmp_buffer.z >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            output_idx -= 1;
            local_code[local_idx] = is_center ? shift : res;
            local_signal[local_idx] = (uint8_t)((output_idx << 1));; 
            local_idx++;

            currValue = static_cast<uint16_t>(tmp_buffer.w & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            output_idx -= 1;
            local_code[local_idx] = is_center ? shift : res;
            local_signal[local_idx] = (uint8_t)((output_idx << 1));; 
            local_idx++;

            currValue = static_cast<uint16_t>((tmp_buffer.w >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            output_idx -= 1;
            local_code[local_idx] = is_center ? shift : res;
            local_signal[local_idx] = (uint8_t)((output_idx << 1));; 
            local_idx++;
        }
    }

    int2* code_int2 = reinterpret_cast<int2*>(code + base_block_start_idx);
    int2* local_code_int2 = reinterpret_cast<int2*>(local_code);

    #pragma unroll
    for (int i = 0; i < cmp_chunk / 8; ++i) {
        code_int2[i] = local_code_int2[i];
    }

    int2* signal_int2 = reinterpret_cast<int2*>(signal + base_block_start_idx);
    int2* local_signal_int2 = reinterpret_cast<int2*>(local_signal);

    #pragma unroll
    for (int i = 0; i < cmp_chunk / 8; ++i) {
        signal_int2[i] = local_signal_int2[i];
    }
     
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

    // printf("block: %d threads: %d\n", gsize, bsize * gsize);

    // 分配设备内存
    void* d_data;
    void* d_decmpdata;
    uint8_t* d_code;
    uint8_t* d_signal;
    int* d_center;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_center, gsize * sizeof(int));
    if (data_type == "uint16") {
        cudaMalloc(&d_decmpdata, bsize * gsize * cmp_chunk * sizeof(uint16_t));
        cudaMalloc(&d_code, bsize * gsize * cmp_chunk * sizeof(uint8_t));  // 假设每个线程最多生成10字节
        cudaMalloc(&d_signal, bsize * gsize * cmp_chunk * sizeof(uint8_t));
    } else if (data_type == "uint32") {
        // cudaMalloc(&d_decmpdata, bsize * gsize * cmp_chunk * sizeof(uint32_t));
        // cudaMalloc(&d_result, bsize * gsize * cmpbytes_per_element_32b * cmp_chunk * sizeof(uint8_t));  // 假设每个线程最多生成10字节
        // cudaMalloc(&d_output, bsize * gsize * cmpbytes_per_element_32b * cmp_chunk * sizeof(uint8_t));
    }
    
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

    // warmup
    for(int i = 0; i < 2; i++)
    {
        if (data_type == "uint16") {
            map_values_kernel_16b<<<gsize, bsize>>>(
                (uint16_t*)d_data, d_code, d_signal, d_center, num_elements, shift);
        } else if (data_type == "uint32") {
            // map_values_kernel_32b<<<gsize, bsize>>>(
            //     (uint32_t*)d_data, d_result, nullptr, d_output_lengths, d_center, num_elements, shift, signal);
        }
    }

    // 设置CUDA事件用于测量时间
    cudaEvent_t start, stop, center_start, center_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&center_start);
    cudaEventCreate(&center_stop);

    // // 启动内核前记录开始时间
    cudaEventRecord(start);

    if (data_type == "uint16") {
        map_values_kernel_16b<<<gsize, bsize>>>(
            (uint16_t*)d_data, d_code, d_signal, d_center, num_elements, shift);
    } else if (data_type == "uint32") {
        // map_values_kernel_32b<<<gsize, bsize>>>(
        //     (uint32_t*)d_data, d_result, nullptr, d_output_lengths,  d_center, num_elements, shift, signal);
    }

    cudaDeviceSynchronize();


    // 结束时间记录
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算压缩运行时间
    float cmp_milliseconds = 0;
    cudaEventElapsedTime(&cmp_milliseconds, start, stop);

    uint8_t* h_code = new uint8_t[num_elements];
    uint8_t* h_signal = new uint8_t[num_elements];

    cudaMemcpy(static_cast<uint8_t*>(h_code), d_code, num_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(static_cast<uint8_t*>(h_signal), d_signal, num_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    int start_idx = 450;
    for(int i = start_idx;i < start_idx + 16; i++)
    {
        printf("%d\t", static_cast<uint16_t*>(h_data)[i]);;
    }
    printf("\n");

    for(int i = start_idx;i < start_idx + 16; i++)
    {
        printf("%d\t", h_code[i]);;
    }
    printf("\n");

    for(int i = start_idx;i < start_idx + 16; i++)
    {
        printf("%d\t", h_signal[i]);;
    }
    printf("\n");

    // 保存结果到文件
    char code_file[1024];
    char signal_file[1024];
    const char* code_name = "code.bin";
    const char* signal_name = "signal.bin";
    snprintf(code_file, 1024, "%s%s", output_file, code_name);
    snprintf(signal_file, 1024, "%s%s", output_file, signal_name);

    save_file(code_file, h_code, num_elements);
    save_file(signal_file, h_signal, num_elements);

    // 计算内核吞吐量 (数据量 / 时间)
    float cmp_throughput = (data_size / 1024.0 / 1024 / 1024) / (cmp_milliseconds / 1000.0f);  // 单位：字节每秒
    // float decmp_throughput = (data_size / 1024.0 / 1024 / 1024) / (decmp_milliseconds / 1000.0f);
    // float cr = data_size * 1.0f / (total_output_size + gsize * bsize / decmp_chunk);
    printf("Compression Kernel throughput: %.2f GB/s\n", cmp_throughput);
    // printf("Decompression Kernel throughput: %.2f GB/s\n", decmp_throughput);
    // printf("Mapping CR: %.2f\n", cr);

    printf("Compression cost %.2f ms\n",cmp_milliseconds);
    // printf("Compute Center cost %.2f ms, take %.2f %\n", center_milliseconds, center_milliseconds / cmp_milliseconds * 100);
    // printf("Decompression cost %.2f ms\n",decmp_milliseconds);

    // 清理内存
    delete[] h_code;
    delete[] h_signal;
    cudaFree(d_data);
    cudaFree(d_code);
    cudaFree(d_signal);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}