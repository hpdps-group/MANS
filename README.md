# MANS: Optimizing ANS Encoding for Multi-Byte Integer Data on the GPU

MANS is a GPU-based ANS compressor for multi-byte integer data, which achieves high compression ratios and high throughput on small multi-byte integer datasets. The core concept of MANS is the ADM kernel(adapative data mapping). We evaluate the performance of MANS using six integer datasets on an A100 GPU. Results demonstrate that MANS achieves compression ratios that are 1.17$\times$ to 2.23$\times$ higher than the original ANS. At the same time,  MANS achieves compression ratios up to 1.92$\times$ higher than state-of-the-art optimized ANS(ADT-FSE).

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
