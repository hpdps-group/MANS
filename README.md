# Multibyte-ANS

Developer: Jinwu Yang@ Institute of Computing Technology, Chinese Academy of Sciences

Multibyte-ANS is a universal byte-range-based ANS (Asymmetric Numeral Systems) entropy encoder and decoder, which can operate at approximately 50-150 GB/s throughput for reasonable data sizes on the MI100/MI210 GPU. It is the GPU version of Yann Collet's [FSE (Finite State Entropy)](https://github.com/Cyan4973/FiniteStateEntropy) ANS library. Currently, the API is only available in C++ (using raw device pointers).

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
 
## License

Multibyte-ANS is licensed with the MIT license, available in the LICENSE file at the top level.