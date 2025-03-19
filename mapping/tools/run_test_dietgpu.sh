#!/bin/bash

# 配置路径
DIETGPU_PATH=/hwj/dietgpu/build/bin
ADM_COMMAND=/hwj/mans/mappingV12
TEST_DIR=/hwj/data/mans-data
filesize_set_KBs="65536 16384 4096 1024 256 64 16 4"
OUTPUT_DIR=./output

# 运行测试
run() {
    mkdir -p $OUTPUT_DIR
    output_file=$OUTPUT_DIR/compression_results-gADM-dietgpu.txt
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
                        encode_dir="/hwj/data/ans_toencode"
                        mkdir -p $split_dir
                        mkdir -p $encode_dir
                        split -b $filesize_set "$file_path" "$split_dir/piece"

                        total_original_size=0
                        total_dietgpu_size=0
                        total_adm_size=0
                        total_dietgpu_adm_size=0

                        piece_count=0
                        for piece in `ls $split_dir`
                        do
                            if [[ $piece_count -ge 10 ]]; then
                                break
                            fi
                            
                            piece_path="$split_dir/$piece"
                            piece_size=$(stat -c%s "$piece_path")
                            total_original_size=$((total_original_size + piece_size))

                            # Step 1: 运行 dietgpu 压缩原始数据
                            cp "$piece_path" "$encode_dir"
                            DIETGPU_OUTPUT=$(cd $DIETGPU_PATH && ./ans_test | grep "bytes ->" | awk '{print $4}' | tr -d ',')
                            dietgpu_size=$(echo "$DIETGPU_OUTPUT" | tail -1)
                            total_dietgpu_size=$((total_dietgpu_size + dietgpu_size))

                            # Step 2: 运行 ADM 处理
                            adm_output_path="$piece_path.adm"
                            $ADM_COMMAND "$piece_path" "$adm_output_path" uint16
                            adm_size=$(stat -c%s "$adm_output_path")
                            total_adm_size=$((total_adm_size + adm_size))

                            # Step 3: 运行 dietgpu 压缩 ADM 处理后数据
                            cp "$adm_output_path" "$encode_dir"
                            DIETGPU_ADM_OUTPUT=$(cd $DIETGPU_PATH && ./ans_test | grep "bytes ->" | awk '{print $4}' | tr -d ',')
                            dietgpu_adm_size=$(echo "$DIETGPU_ADM_OUTPUT" | tail -1)
                            total_dietgpu_adm_size=$((total_dietgpu_adm_size + dietgpu_adm_size))

                            # 清理临时文件
                            rm -f "$adm_output_path"

                            piece_count=$((piece_count + 1))
                        done

                        # 计算压缩比
                        cr_dietgpu=$(echo "scale=3; $total_original_size / $total_dietgpu_size" | bc)
                        cr_adm=$(echo "scale=3; $total_original_size / $total_adm_size" | bc)
                        cr_dietgpu_adm=$(echo "scale=3; $total_original_size / $total_dietgpu_adm_size" | bc)

                        # 记录结果
                        echo "SIZE: $filesize_set_KB KB" >> $output_file
                        echo "  Original Size: $total_original_size Bytes" >> $output_file
                        echo "  DietGPU Compressed Size: $total_dietgpu_size Bytes (CR: $cr_dietgpu)" >> $output_file
                        echo "  ADM Processed Size: $total_adm_size Bytes (CR: $cr_adm)" >> $output_file
                        echo "  DietGPU After ADM Size: $total_dietgpu_adm_size Bytes (CR: $cr_dietgpu_adm)" >> $output_file
                        echo "" >> $output_file

                        # 删除分片目录
                        rm -rf $split_dir
                        rm -rf $encode_dir
                    done
                fi
            done
        fi
    done

    echo "FINISHED." >> $output_file
}

# 运行测试
run
