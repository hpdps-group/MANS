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

__global__ void decompress_kernel_16b(uint8_t* compressed_data, const int* offsets, const int* output_lengths, uint16_t* decompressed_data, int* centers, int shift, uint8_t signal) {
    const int block_id = blockIdx.x * blockDim.x + threadIdx.x;  // 每个线程对应一个 block
    const int lane = block_id & 0x1f;
    const int bid = blockIdx.x;

    // 获取当前 block 的起始偏移和压缩数据段长度
    int start_offset = offsets[block_id * decmp_chunk];    // 每隔 decmp_chunk 个读取
    int length = 0;
    for (int i = 0; i < decmp_chunk; i++) {
        length += output_lengths[block_id * decmp_chunk + i];  // 累加 decmp_chunk 个线程的长度
    }
    // int end_offset = start_offset + length;  // 当前 block 的压缩数据结束位置
    int decompressed_idx = block_id * cmp_chunk * decmp_chunk;

    uint8_t local_cmp_data[cmp_chunk * cmpbytes_per_element_16b * decmp_chunk];

    int count = length / aligned;
    int2* local_int2 = reinterpret_cast<int2*>(local_cmp_data);
    int2* cmp_int2 = reinterpret_cast<int2*>(compressed_data + start_offset);

    #pragma unroll
    for (int i = 0; i < count; ++i) {
        local_int2[i] = cmp_int2[i];
    } 

    uint16_t local_result[cmp_chunk * decmp_chunk];

    int  center = lane < 16 ? centers[bid * 2] : centers[bid * 2 + 1];

    // 逐字节解码当前 block 的压缩数据段
    int read_pos = 0;
    int out_pos = 0;
    while (read_pos < length) {
        uint8_t current_byte = local_cmp_data[read_pos];

        if(current_byte < 254)
        {
            int signal_count = 0;
            
            signal_count = local_cmp_data[read_pos] != 0 ? 0
                                                         : local_cmp_data[read_pos + 1] != 0 ? 1
                                                         : local_cmp_data[read_pos + 2] != 0 ? 2 : 3;

            read_pos += signal_count;
            uint8_t code = local_cmp_data[read_pos];
            int diff = (code % 2 == 1) ? ((code - 1) / 2) : ((code) / 2);
            diff += signal_count * 126;
            local_result[out_pos] = (code % 2 == 1) ? center - diff : center + diff;
            read_pos++;
            out_pos += 1;

        }
        else
        {
            local_result[out_pos] = current_byte == 254 ? (local_cmp_data[read_pos + 1] << 8) | local_cmp_data[read_pos + 2] : local_result[out_pos];
            read_pos = current_byte == 254 ? read_pos + 3 : (read_pos + 8) / 8 * 8;
            out_pos =  current_byte == 254 ? out_pos + 1 : out_pos;
        }
    }

    int4* local_result_int4 = reinterpret_cast<int4*>(local_result);
    int4* decmp_int4 = reinterpret_cast<int4*>(decompressed_data + decompressed_idx);
    count = cmp_chunk * decmp_chunk / 8;

    #pragma unroll
    for (int i = 0; i < count; ++i) {
        decmp_int4[i] = local_result_int4[i];
    }  
}

__global__ void decompress_kernel_32b(uint8_t* compressed_data, const int* offsets, const int* output_lengths, uint32_t* decompressed_data, int* centers, int shift, uint8_t signal) {
    const int block_id = blockIdx.x * blockDim.x + threadIdx.x;  // 每个线程对应一个 block
    const int lane = block_id & 0x1f;
    const int bid = blockIdx.x;

    // 获取当前 block 的起始偏移和压缩数据段长度
    int start_offset = offsets[block_id * decmp_chunk];    // 每隔 decmp_chunk 个读取
    int length = 0;
    for (int i = 0; i < decmp_chunk; i++) {
        length += output_lengths[block_id * decmp_chunk + i];  // 累加 decmp_chunk 个线程的长度
    }
    // int end_offset = start_offset + length;  // 当前 block 的压缩数据结束位置
    int decompressed_idx = block_id * cmp_chunk * decmp_chunk;

    uint8_t local_cmp_data[cmp_chunk * cmpbytes_per_element_32b * decmp_chunk];

    int count = length / aligned;
    int2* local_int2 = reinterpret_cast<int2*>(local_cmp_data);
    int2* cmp_int2 = reinterpret_cast<int2*>(compressed_data + start_offset);

    #pragma unroll
    for (int i = 0; i < count; ++i) {
        local_int2[i] = cmp_int2[i];
    } 

    uint32_t local_result[cmp_chunk * decmp_chunk];

    int  center = lane < 16 ? centers[bid * 2] : centers[bid * 2 + 1];


    // 逐字节解码当前 block 的压缩数据段
    int read_pos = 0;
    int out_pos = 0;
    while (read_pos < length) {
        uint8_t current_byte = local_cmp_data[read_pos];

        if(current_byte < 254)
        {
            int signal_count = 0;
            
            signal_count = local_cmp_data[read_pos] != 0 ? 0
                                                         : local_cmp_data[read_pos + 1] != 0 ? 1
                                                         : local_cmp_data[read_pos + 2] != 0 ? 2
                                                         : local_cmp_data[read_pos + 3] != 0 ? 3
                                                         : local_cmp_data[read_pos + 4] != 0 ? 4
                                                         : local_cmp_data[read_pos + 5] != 0 ? 5
                                                         : local_cmp_data[read_pos + 6] != 0 ? 6 : 7;

            read_pos += signal_count;
            uint8_t code = local_cmp_data[read_pos];
            int diff = (code % 2 == 1) ? ((code - 1) / 2) : ((code) / 2);
            diff += signal_count * 126;
            local_result[out_pos] = (code % 2 == 1) ? center - diff : center + diff;
            read_pos++;
            out_pos += 1;
        }
        else
        {
            local_result[out_pos] = current_byte == 254 ? (local_cmp_data[read_pos + 1] << 24) | local_cmp_data[read_pos + 2] << 16 | local_cmp_data[read_pos + 3] << 8 | local_cmp_data[read_pos + 4] : local_result[out_pos];
            read_pos = current_byte == 254 ? read_pos + 5 : (read_pos + 8) / 8 * 8;
            out_pos =  current_byte == 254 ? out_pos + 1 : out_pos;
        }
    }

    int4* local_result_int4 = reinterpret_cast<int4*>(local_result);
    int4* decmp_int4 = reinterpret_cast<int4*>(decompressed_data + decompressed_idx);
    count = cmp_chunk * decmp_chunk / 4;

    #pragma unroll
    for (int i = 0; i < count; ++i) {
        decmp_int4[i] = local_result_int4[i];
    }  
}


__global__ void map_values_kernel_16b(const uint16_t* data, uint8_t* result, const int* offsets, int* output_lengths, int* centers, int data_size, int shift, uint8_t signal) 
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
    uint8_t local_result[cmp_chunk * cmpbytes_per_element_16b] = {0};
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

            if(output_idx <= max_signals_16b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }

            currValue = static_cast<uint16_t>((tmp_buffer.x >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            if(output_idx <= max_signals_16b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }

            currValue = static_cast<uint16_t>(tmp_buffer.y & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            if(output_idx <= max_signals_16b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }

            currValue = static_cast<uint16_t>((tmp_buffer.y >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            if(output_idx <= max_signals_16b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }

            currValue = static_cast<uint16_t>(tmp_buffer.z & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            if(output_idx <= max_signals_16b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }

            currValue = static_cast<uint16_t>((tmp_buffer.z >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            if(output_idx <= max_signals_16b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }

            currValue = static_cast<uint16_t>(tmp_buffer.w & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            if(output_idx <= max_signals_16b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }

            currValue = static_cast<uint16_t>((tmp_buffer.w >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            if(output_idx <= max_signals_16b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }
        }
    }

    // 将本线程的输出长度记录到 output_lengths
    int aligned_idx = (local_idx % aligned == 0) ? local_idx / aligned : local_idx / aligned + 1;
    #pragma unroll
    for(int i = local_idx; i < aligned_idx * aligned; i++) 
        local_result[i] = 255;

    output_lengths[idx] = aligned_idx * aligned;
    // output_lengths[idx] = local_idx;

    int global_pos = warp * cmp_tblock_size * cmp_chunk * cmpbytes_per_element_16b;
    
    int2* result_int2 = reinterpret_cast<int2*>(result + global_pos);
    int2* local_result_int2 = reinterpret_cast<int2*>(local_result);

    #pragma unroll
    for (int i = 0; i < aligned_idx; ++i) {
        result_int2[i * 32 + lane] = local_result_int2[i];
    }    
}

__global__ void map_values_kernel_32b(const uint32_t* data, uint8_t* result, const int* offsets, int* output_lengths, int* centers, int data_size, int shift, uint8_t signal) 
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
    uint8_t local_result[cmp_chunk * cmpbytes_per_element_32b] = {0};
    int diff = 0;
    int output_idx = 0;
    int local_idx = 0;
    uint32_t currValue = 0;

    bool is_center;
    uint32_t remain = 0;
    uint8_t res = 0;

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
 
    __syncthreads();

    // 使用 `center[bid]` 继续后续计算
    int center = centers[bid];

    for(int j = 0; j < block_num; j++)
    {
        if(base_block_start_idx > data_size) break;

        // #pragma unroll
        for(int i = base_block_start_idx; i < base_block_end_idx && i < data_size; i+=4)
        {
            tmp_buffer = reinterpret_cast<const uint4*>(data)[i / 4];

            currValue = static_cast<uint32_t>(tmp_buffer.x);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            if(output_idx <= max_signals_32b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 24) & 0xFF;
                local_result[local_idx++] = (currValue >> 16) & 0xFF;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }

            currValue = static_cast<uint32_t>(tmp_buffer.y);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            if(output_idx <= max_signals_32b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 24) & 0xFF;
                local_result[local_idx++] = (currValue >> 16) & 0xFF;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }

            currValue = static_cast<uint32_t>(tmp_buffer.z);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            if(output_idx <= max_signals_32b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 24) & 0xFF;
                local_result[local_idx++] = (currValue >> 16) & 0xFF;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }

            currValue = static_cast<uint32_t>(tmp_buffer.w);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            if(output_idx <= max_signals_32b + 1)
            {
                res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                               : (diff + 126 - output_idx * 126) * 2 + shift;
                // for(int k = 0; k < output_idx - 1; k++)
                // {
                //     local_result[local_idx + k] = signal;
                // }
                local_result[local_idx + output_idx - 1] = is_center ? shift : res;
                local_idx += output_idx;
            }
            else
            {
                local_result[local_idx++] = 254;
                local_result[local_idx++] = (currValue >> 24) & 0xFF;
                local_result[local_idx++] = (currValue >> 16) & 0xFF;
                local_result[local_idx++] = (currValue >> 8) & 0xFF;
                local_result[local_idx++] = currValue & 0xFF;
            }
        }
    }

    // 将本线程的输出长度记录到 output_lengths
    int aligned_idx = (local_idx % aligned == 0) ? local_idx / aligned : local_idx / aligned + 1;
    #pragma unroll
    for(int i = local_idx; i < aligned_idx * aligned; i++) 
        local_result[i] = 255;

    output_lengths[idx] = aligned_idx * aligned;
    // output_lengths[idx] = local_idx;

    int global_pos = warp * cmp_tblock_size * cmp_chunk * cmpbytes_per_element_32b;
    
    int2* result_int2 = reinterpret_cast<int2*>(result + global_pos);
    int2* local_result_int2 = reinterpret_cast<int2*>(local_result);

    #pragma unroll
    for (int i = 0; i < aligned_idx; ++i) {
        result_int2[i * 32 + lane] = local_result_int2[i];
    }    
}

template <typename T>
__global__ void coalescing(uint8_t* result, const int* offsets, int* output_lengths, uint8_t* output, int data_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    
    const int cmpbytes_per_element = sizeof(T) == 2 ? cmpbytes_per_element_16b : cmpbytes_per_element_32b;  // 根据数据类型选择值

    if (idx * cmp_chunk < data_size)
    {
        const int global_pos = offsets[idx];
        const int result_start = warp * cmp_tblock_size * cmp_chunk * cmpbytes_per_element;
        const int length = output_lengths[idx] / aligned;
        int2* output_int2 = reinterpret_cast<int2*>(output + global_pos);
        int2* result_int2 = reinterpret_cast<int2*>(result + result_start);

        #pragma unroll
        for(int i = 0; i < length; i++)
        {
            output_int2[i] = result_int2[i * 32 + lane];
        }
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

template <typename T>
void allocate_memory(void** d_data, int num_elements) {
    cudaMalloc(d_data, num_elements * sizeof(T));
}


// Helper function to save result to binary file
void save_file(const char* filename, const uint8_t* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), size);
    file.close();
}

template <typename T>
int compute_median(void* d_data, int num_elements) {
    // 将原始设备指针封装为 Thrust 设备指针
    thrust::device_ptr<T> dev_data(static_cast<T*>(d_data));

    // 创建设备向量副本，并拷贝原始数据
    thrust::device_vector<T> d_copy(dev_data, dev_data + num_elements);

    // 对副本进行排序
    thrust::sort(d_copy.begin(), d_copy.end());

    return static_cast<int>(d_copy[num_elements / 2]);
}

template <typename T>
int compute_mean(void* d_data, int num_elements) {
    // 将原始设备指针封装为 Thrust 设备指针
    thrust::device_ptr<T> dev_data(static_cast<T*>(d_data));

    // 使用 Thrust 并行归约计算总和
    long long total_sum = thrust::reduce(dev_data, dev_data + num_elements, static_cast<long long>(0), thrust::plus<long long>());

    // 计算平均值
    return static_cast<int>(static_cast<double>(total_sum) / num_elements);
}


int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input file> <output file> <data type>" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    std::string data_type = argv[3]; // 从命令行获取数据类型
    int center = 0;
    bool has_center = false;

    if(argc == 5)
    {
        center = std::stoi(argv[4]);
        has_center = true;
    }
    int shift = 1;
    uint8_t signal = 0;

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
    uint8_t* d_result;
    uint8_t* d_output;
    int* d_output_lengths;
    int* d_offsets;
    int* d_center;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_center, gsize * sizeof(int));
    if (data_type == "uint16") {
        cudaMalloc(&d_decmpdata, bsize * gsize * cmp_chunk * sizeof(uint16_t));
        cudaMalloc(&d_result, bsize * gsize * cmpbytes_per_element_16b * cmp_chunk * sizeof(uint8_t));  // 假设每个线程最多生成10字节
        cudaMalloc(&d_output, bsize * gsize * cmpbytes_per_element_16b * cmp_chunk * sizeof(uint8_t));
    } else if (data_type == "uint32") {
        cudaMalloc(&d_decmpdata, bsize * gsize * cmp_chunk * sizeof(uint32_t));
        cudaMalloc(&d_result, bsize * gsize * cmpbytes_per_element_32b * cmp_chunk * sizeof(uint8_t));  // 假设每个线程最多生成10字节
        cudaMalloc(&d_output, bsize * gsize * cmpbytes_per_element_32b * cmp_chunk * sizeof(uint8_t));
    }
    
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_output_lengths, bsize * gsize * sizeof(int));
    cudaMalloc(&d_offsets, bsize * gsize * sizeof(int));

    // printf("%d %d\n", bsize, gsize);

    // 如果没有提供 center，则计算中位数
    // if (!has_center) {
    //     center = compute_center<uint16_t>(d_data, num_elements);
    //     std::cout << "Computed center (median): " << center << std::endl;
    //     //center = 1541;
    //     has_center = true;
    // }
    center = 1541;

    // 使用 thrust 计算前缀和
    thrust::device_ptr<int> dev_output_lengths(d_output_lengths);
    thrust::device_ptr<int> dev_offsets(d_offsets);


    // warmup
    for(int i = 0; i < 2; i++)
    {
        if (data_type == "uint16") {
            map_values_kernel_16b<<<gsize, bsize>>>(
                (uint16_t*)d_data, d_result, nullptr, d_output_lengths, d_center, num_elements, shift, signal);
        } else if (data_type == "uint32") {
            map_values_kernel_32b<<<gsize, bsize>>>(
                (uint32_t*)d_data, d_result, nullptr, d_output_lengths, d_center, num_elements, shift, signal);
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
    // cudaEventRecord(center_start);


    // if (data_type == "uint16" && has_center == false) {
    //     center = compute_mean<uint16_t>(d_data, num_elements * 0.1);
    // } else if (data_type == "uint32" && has_center == false) {
    //     center = compute_mean<uint32_t>(d_data, num_elements);
    // }

    // cudaEventRecord(center_stop);
    // cudaEventSynchronize(center_stop);

    // printf("center: %d\n", center);

    // 启动内核，计算每个线程的输出长度
    if (data_type == "uint16") {
        map_values_kernel_16b<<<gsize, bsize>>>(
            (uint16_t*)d_data, d_result, nullptr, d_output_lengths, d_center, num_elements, shift, signal);
    } else if (data_type == "uint32") {
        map_values_kernel_32b<<<gsize, bsize>>>(
            (uint32_t*)d_data, d_result, nullptr, d_output_lengths,  d_center, num_elements, shift, signal);
    }

    cudaDeviceSynchronize();


    thrust::exclusive_scan(dev_output_lengths, dev_output_lengths + bsize * gsize, dev_offsets);

    cudaDeviceSynchronize();

    if (data_type == "uint16") {
        coalescing<uint16_t><<<gsize, bsize>>>(d_result, d_offsets, d_output_lengths, d_output, num_elements);
    } else if (data_type == "uint32") {
        coalescing<uint32_t><<<gsize, bsize>>>(d_result, d_offsets, d_output_lengths, d_output, num_elements);
    }

    cudaDeviceSynchronize();

    // 结束时间记录
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算压缩运行时间
    float cmp_milliseconds = 0;
    cudaEventElapsedTime(&cmp_milliseconds, start, stop);

    float center_milliseconds = 0;
    cudaEventElapsedTime(&center_milliseconds, center_start, center_stop);

    cudaEventRecord(start);

    bsize = 32;
    int part = decmp_chunk * bsize / 32;
    int gsize2 = (gsize + part - 1) / part;

    // printf("grid size %d\n", gsize2);

    if (data_type == "uint16") {
        decompress_kernel_16b<<<gsize2, bsize>>>(d_output, d_offsets, d_output_lengths, (uint16_t*)d_decmpdata, d_center, shift, signal);
    } else if (data_type == "uint32") {
        decompress_kernel_32b<<<gsize2, bsize>>>(d_output, d_offsets, d_output_lengths, (uint32_t*)d_decmpdata, d_center, shift, signal);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算解压运行时间
    float decmp_milliseconds = 0;
    cudaEventElapsedTime(&decmp_milliseconds, start, stop);

    // 复制偏移量和输出长度到主机
    int* h_offsets = new int[bsize * gsize];
    int* h_output_lengths = new int[bsize * gsize];
    cudaMemcpy(h_offsets, d_offsets, bsize * gsize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_lengths, d_output_lengths, bsize * gsize * sizeof(int), cudaMemcpyDeviceToHost);

    // 计算总的输出大小
    int total_output_size = h_offsets[bsize * gsize - 1] + h_output_lengths[bsize * gsize - 1];

    // printf("%d\n", h_output_lengths[0]);

    // 分配主机内存并复制结果
    uint8_t* h_output = new uint8_t[total_output_size];
    void* h_decmpdata = nullptr;

    if (data_type == "uint16") {
        h_decmpdata = new uint16_t[num_elements];
    } else if (data_type == "uint32") {
        h_decmpdata = new uint32_t[num_elements];
    }
    cudaMemcpy(h_output, d_output, total_output_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (data_type == "uint16") {
        cudaMemcpy(static_cast<uint16_t*>(h_decmpdata), d_decmpdata, num_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    } else if (data_type == "uint32") {
        cudaMemcpy(static_cast<uint32_t*>(h_decmpdata), d_decmpdata, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    // printf("decmpdata0: %d\n", static_cast<uint16_t*>(h_decmpdata)[0]);

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

    // for(int i = 0;i < 16; i++)
    // {
    //     printf("%d\t", static_cast<uint16_t*>(h_data)[i]);;
    // }
    // printf("\n");

    // for(int i = 0;i < 16; i++)
    // {
    //     printf("%d\t", h_output[i]);;
    // }
    // printf("\n");

    // for(int i = 0;i < 16; i++)
    // {
    //     printf("%d\t", static_cast<uint16_t*>(h_decmpdata)[i]);;
    // }
    // printf("\n");

    // 保存结果到文件
    save_file(output_file, h_output, total_output_size);

    // 计算内核吞吐量 (数据量 / 时间)
    float cmp_throughput = (data_size / 1024.0 / 1024 / 1024) / (cmp_milliseconds / 1000.0f);  // 单位：字节每秒
    float decmp_throughput = (data_size / 1024.0 / 1024 / 1024) / (decmp_milliseconds / 1000.0f);
    float cr = data_size * 1.0f / (total_output_size + gsize * bsize / decmp_chunk);
    printf("Compression Kernel throughput: %.2f GB/s\n", cmp_throughput);
    printf("Decompression Kernel throughput: %.2f GB/s\n", decmp_throughput);
    printf("Mapping CR: %.2f\n", cr);

    printf("Compression cost %.2f ms\n",cmp_milliseconds);
    printf("Compute Center cost %.2f ms, take %.2f %\n", center_milliseconds, center_milliseconds / cmp_milliseconds * 100);
    printf("Decompression cost %.2f ms\n",decmp_milliseconds);

    // 清理内存
    delete[] h_output;
    delete[] h_output_lengths;
    delete[] h_offsets;
    cudaFree(d_data);
    cudaFree(d_decmpdata);
    cudaFree(d_result);
    cudaFree(d_output);
    cudaFree(d_output_lengths);
    cudaFree(d_offsets);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}