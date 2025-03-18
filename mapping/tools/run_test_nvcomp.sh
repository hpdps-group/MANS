#!/bin/bash

# 配置路径
NVCOMP_PATH=/home/gyd/HWJ/nvcomp/bin
ADM_COMMAND=/home/gyd/HWJ/mans/mappingV12
TEST_DIR=/home/gyd/HWJ/data/mans-data
filesize_set_KBs="65536 16384 4096 1024 256 64 16 4"
OUTPUT_DIR=./output

# 运行测试
run() {
    mkdir -p $OUTPUT_DIR
    output_file=$OUTPUT_DIR/compression_results-gADM-gNVCOMP.txt
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

                        total_original_size=0
                        total_gdeflate_size=0
                        total_ans_size=0
                        total_adm_size=0
                        total_gdeflate_adm_size=0
                        total_ans_adm_size=0

                        piece_count=0
                        for piece in `ls $split_dir`
                        do
                            if [[ $piece_count -ge 10 ]]; then
                                break
                            fi
                            
                            piece_path="$split_dir/$piece"
                            piece_size=$(stat -c%s "$piece_path")
                            total_original_size=$((total_original_size + piece_size))

                            # Step 1: GDeflate 压缩
                            gdeflate_output=$($NVCOMP_PATH/benchmark_gdeflate_chunked -f "$piece_path" | grep "comp_size" | awk '{print $2}' | tr -d ',')
                            total_gdeflate_size=$((total_gdeflate_size + gdeflate_output))

                            # Step 2: ANS 压缩
                            ans_output=$($NVCOMP_PATH/benchmark_ans_chunked -f "$piece_path" | grep "comp_size" | awk '{print $2}' | tr -d ',')
                            total_ans_size=$((total_ans_size + ans_output))

                            # Step 3: ADM 处理
                            adm_output_path="$piece_path.adm"
                            $ADM_COMMAND "$piece_path" "$adm_output_path" uint16
                            adm_size=$(stat -c%s "$adm_output_path")
                            total_adm_size=$((total_adm_size + adm_size))

                            # Step 4: GDeflate 压缩 ADM 输出
                            gdeflate_adm_output=$($NVCOMP_PATH/benchmark_gdeflate_chunked -f "$adm_output_path" | grep "comp_size" | awk '{print $2}' | tr -d ',')
                            total_gdeflate_adm_size=$((total_gdeflate_adm_size + gdeflate_adm_output))

                            # Step 5: ANS 压缩 ADM 输出
                            ans_adm_output=$($NVCOMP_PATH/benchmark_ans_chunked -f "$adm_output_path" | grep "comp_size" | awk '{print $2}' | tr -d ',')
                            total_ans_adm_size=$((total_ans_adm_size + ans_adm_output))

                            # 清理临时文件
                            rm -f "$adm_output_path"

                            piece_count=$((piece_count + 1))
                        done

                        # 计算压缩比
                        cr_gdeflate=$(echo "scale=3; $total_original_size / $total_gdeflate_size" | bc)
                        cr_ans=$(echo "scale=3; $total_original_size / $total_ans_size" | bc)
                        cr_adm=$(echo "scale=3; $total_original_size / $total_adm_size" | bc)
                        cr_gdeflate_adm=$(echo "scale=3; $total_original_size / $total_gdeflate_adm_size" | bc)
                        cr_ans_adm=$(echo "scale=3; $total_original_size / $total_ans_adm_size" | bc)

                        # 记录结果
                        echo "SIZE: $filesize_set_KB KB" >> $output_file
                        echo "  Original Size: $total_original_size Bytes" >> $output_file
                        echo "  GDeflate Compressed Size: $total_gdeflate_size Bytes (CR: $cr_gdeflate)" >> $output_file
                        echo "  ANS Compressed Size: $total_ans_size Bytes (CR: $cr_ans)" >> $output_file
                        echo "  ADM Processed Size: $total_adm_size Bytes (CR: $cr_adm)" >> $output_file
                        echo "  GDeflate After ADM Size: $total_gdeflate_adm_size Bytes (CR: $cr_gdeflate_adm)" >> $output_file
                        echo "  ANS After ADM Size: $total_ans_adm_size Bytes (CR: $cr_ans_adm)" >> $output_file
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
