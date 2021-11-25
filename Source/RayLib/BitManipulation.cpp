#include "BitManipulation.h"
#include "System.h"

uint64_t Utility::FindLastSet64(uint64_t val)
{
    #ifdef METURAY_WIN
        unsigned long ul;
        _BitScanReverse64(&ul, val);
        return ul;
    #elif defined(METURAY_LINUX)
        return (sizeof(uint64_t) * BYTE_BITS) - __builtin_clzl(val) - 1;
    #endif
}

uint32_t Utility::FindLastSet32(uint32_t val)
{
    #ifdef METURAY_WIN
        unsigned long ul;
        _BitScanReverse(&ul, val);
        return ul;
    #elif defined(METURAY_LINUX)
        return (sizeof(uint32_t) * BYTE_BITS) - __builtin_clz(val) - 1;
    #endif
}