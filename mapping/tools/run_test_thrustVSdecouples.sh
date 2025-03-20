#!/bin/bash

# 配置路径
THRUST=/home/gyd/HWJ/mans/mappingV12
DECOUPLED=/home/gyd/HWJ/mans/mappingV13
TEST_DIR=/home/gyd/HWJ/data/mans-data
OUTPUT_DIR=./output
filesize_set_KBs="65536 16384 4096 1024 256 64 16 4"

# 运行测试
run() {
    mkdir -p $OUTPUT_DIR
    output_file=$OUTPUT_DIR/throughput_thrustVSdecoupled_results.txt
    echo "START" > $output_file

    for dir in `ls $TEST_DIR`
    do
        if [[ -d $TEST_DIR"/"$dir ]]; then
            echo "Processing Directory: $dir"
            echo "DIR: $dir" >> $output_file

            for file in `ls $TEST_DIR"/"$dir`
            do
                if [[ $file == *".u2" ]]; then
                    echo "   FILE: $file"
                    echo "FILE: $file" >> $output_file
                    file_path="$TEST_DIR/$dir/$file"
                    file_size_original=$(stat -c%s "$file_path")

                    for filesize_set_KB in $filesize_set_KBs
                    do
                        filesize_set=$((filesize_set_KB * 1024))

                        if [[ $file_size_original -lt $filesize_set ]]; then
                            echo "  Skipping size: $filesize_set, larger than file"
                            continue
                        fi

                        # 生成分片
                        split_dir="./split_files"
                        mkdir -p $split_dir
                        split -b $filesize_set "$file_path" "$split_dir/piece"

                        total_throughput_v12=0
                        total_throughput_v13=0
                        count=0

                        for piece in `ls $split_dir`
                        do
                           if [[ $count -ge 10 ]]; then
                                break
                            fi
                            piece_path="$split_dir/$piece"
                            output_path="$piece_path.out"

                            # 运行 mappingV12 并提取 Total Cmp throughput
                            throughput_v12=$($THRUST "$piece_path" "$output_path" uint16 | grep "Total Cmp throughput" | awk '{print $4}')
                            if [[ ! -z "$throughput_v12" ]]; then
                                total_throughput_v12=$(echo "$total_throughput_v12 + $throughput_v12" | bc)
                            fi

                            # 运行 mappingV13 并提取 ADM Kernel throughput
                            throughput_v13=$($DECOUPLED "$piece_path" "$output_path" uint16 | grep "ADM Kernel throughput" | awk '{print $4}')
                            if [[ ! -z "$throughput_v13" ]]; then
                                total_throughput_v13=$(echo "$total_throughput_v13 + $throughput_v13" | bc)
                            fi

                            count=$((count + 1))
                        done

                        # 计算平均吞吐量
                        avg_throughput_v12=$(echo "scale=3; $total_throughput_v12 / $count" | bc)
                        avg_throughput_v13=$(echo "scale=3; $total_throughput_v13 / $count" | bc)

                        # 记录结果
                        echo "SIZE: $filesize_set_KB KB" >> $output_file
                        echo "  thrust Avg Throughput: $avg_throughput_v12 GB/s" >> $output_file
                        echo "  decoupled Avg Throughput: $avg_throughput_v13 GB/s" >> $output_file
                        echo "" >> $output_file

                        # 删除分片目录
                        rm -rf $split_dir
                    done
                fi
            done
        fi
    done

    echo "FINISHED." >> $output_file
}

# 运行测试
run
