#pragma once
#include <cstdint>

namespace mans {


struct  MansParams {
    uint32_t backend;       // 0: CPU, 1: GPU
    uint32_t dtype;         // 0: U16, 1: U32
    uint32_t adm_threshold; // (block max diff > adm_threshold) -> skip adm mode
};
static_assert(sizeof(MansParams) % 4 == 0, "MansParams size must be multiple of 4 bytes");

namespace Backend {
    constexpr uint32_t CPU = 0;
    constexpr uint32_t NVIDIA = 1;
}

namespace DataType {
    constexpr uint32_t U16 = 0;
    constexpr uint32_t U32 = 1;
}

// === 2. 文件头定义 ===
struct MansHeader {
    std::uint8_t codec;  // 1 = ADM, 2 = ANS
};
// 静态断言：确保编译器不会给它加 padding，保证它占 1 字节
static_assert(sizeof(MansHeader) == 1, "MansHeader must be 1 byte");

} // namespace mans