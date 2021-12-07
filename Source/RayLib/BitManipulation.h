#pragma once

#include <cstdint>
#include <bitset>
#include "Types.h"

namespace Utility
{
    uint64_t FindLastSet64(uint64_t);
    uint32_t FindLastSet32(uint32_t);

    uint32_t NextPowOfTwo(uint32_t);
    uint64_t NextPowOfTwo(uint64_t);

    template<class T>
    uint32_t BitCount(T);
}

template<class T>
uint32_t Utility::BitCount(T val)
{
    return static_cast<uint32_t>(std::bitset<sizeof(T) * BYTE_BITS>(val).count());
}