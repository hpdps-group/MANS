# MANS: Optimizing ANS Encoding for Multi-Byte Integer Data on NVIDIA GPUs

(C) 2025 by Institute of Computing Technology, Chinese Academy of Sciences. 
- Developer: Wenjing Huang, Jinwu Yang 
- Advisor: Dingwen Tao, Guangming Tan

## Building

Clone this repo using

```shell
git clone https://github.com/ewTomato/Multibyte-ANS.git
```

Do the standard CMake thing:

```shell
cd Multibyte-ANS; mkdir build; cd build;
sudo cmake --build .
```

## Instructions for Use

input_path: the path of the data file to use for compression

temp_path: the path to the file used to store the compressed data

output_path: the path to the file used to store the extracted data

```shell
cd Multibyte-ANS; cd build;
./bin/compress "input_path" "temp_path" "output_path"
```

## License

Multibyte-ANS is licensed with the MIT license, available in the LICENSE file at the top level.
