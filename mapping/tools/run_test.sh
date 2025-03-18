#!/bin/bash

# 配置路径
FSE_PATH=/home/gyd/HWJ/FiniteStateEntropy-dev
ADM_COMMAND=/home/gyd/HWJ/mans/mappingV12
TEST_DIR=/home/gyd/HWJ/data/mans-data
filesize_set_KBs="65536 16384 4096 1024 256 64 16 4"
OUTPUT_DIR=./output

# 运行测试
run() {
    mkdir -p $OUTPUT_DIR
    output_file=$OUTPUT_DIR/compression_results.txt
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
                        total_fse_hf_size=0
                        total_fse_f_size=0
                        total_adm_size=0
                        total_fse_hf_adm_size=0
                        total_fse_f_adm_size=0

                        piece_count=0
                        for piece in `ls $split_dir`
                        do
                            if [[ $piece_count -ge 10 ]]; then
                                break
                            fi
                            
                            piece_path="$split_dir/$piece"
                            piece_size=$(stat -c%s "$piece_path")
                            total_original_size=$((total_original_size + piece_size))

                            # Step 1: FSE -hf 压缩
                            $FSE_PATH/fse -hf "$piece_path"
                            piece_hf_size=$(stat -c%s "$piece_path.fse")
                            total_fse_hf_size=$((total_fse_hf_size + piece_hf_size))

                            # Step 2: FSE -f 压缩
                            $FSE_PATH/fse -f "$piece_path"
                            piece_f_size=$(stat -c%s "$piece_path.fse")
                            total_fse_f_size=$((total_fse_f_size + piece_f_size))

                            # Step 3: ADM 处理
                            adm_output_path="$piece_path.adm"
                            $ADM_COMMAND "$piece_path" "$adm_output_path" uint16
                            adm_size=$(stat -c%s "$adm_output_path")
                            total_adm_size=$((total_adm_size + adm_size))

                            # Step 4: FSE -hf 压缩 ADM 输出
                            $FSE_PATH/fse -hf "$adm_output_path"
                            adm_hf_size=$(stat -c%s "$adm_output_path.fse")
                            total_fse_hf_adm_size=$((total_fse_hf_adm_size + adm_hf_size))

                            # Step 5: FSE -f 压缩 ADM 输出
                            $FSE_PATH/fse -f "$adm_output_path"
                            adm_f_size=$(stat -c%s "$adm_output_path.fse")
                            total_fse_f_adm_size=$((total_fse_f_adm_size + adm_f_size))

                            # 清理临时文件
                            rm -f "$piece_path.fse" "$adm_output_path" "$adm_output_path.fse"

                            piece_count=$((piece_count + 1))
                        done

                        # 计算压缩比
                        cr_fse_hf=$(echo "scale=3; $total_original_size / $total_fse_hf_size" | bc)
                        cr_fse_f=$(echo "scale=3; $total_original_size / $total_fse_f_size" | bc)
                        cr_adm=$(echo "scale=3; $total_original_size / $total_adm_size" | bc)
                        cr_fse_hf_adm=$(echo "scale=3; $total_original_size / $total_fse_hf_adm_size" | bc)
                        cr_fse_f_adm=$(echo "scale=3; $total_original_size / $total_fse_f_adm_size" | bc)

                        # 记录结果
                        echo "SIZE: $filesize_set_KB KB" >> $output_file
                        echo "  Original Size: $total_original_size Bytes" >> $output_file
                        echo "  FSE -hf Compressed Size: $total_fse_hf_size Bytes (CR: $cr_fse_hf)" >> $output_file
                        echo "  FSE -f Compressed Size: $total_fse_f_size Bytes (CR: $cr_fse_f)" >> $output_file
                        echo "  ADM Processed Size: $total_adm_size Bytes (CR: $cr_adm)" >> $output_file
                        echo "  FSE -hf After ADM Size: $total_fse_hf_adm_size Bytes (CR: $cr_fse_hf_adm)" >> $output_file
                        echo "  FSE -f After ADM Size: $total_fse_f_adm_size Bytes (CR: $cr_fse_f_adm)" >> $output_file
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
