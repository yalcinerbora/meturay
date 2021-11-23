#include "BitManipulation.h"
#include "System.h"

uint64_t Utility::FindLastSet64(uint64_t val)
{
    #ifdef METURAY_WIN
        unsigned long ul;
        _BitScanReverse64(&ul, val);
        return ul;
    #elif defined(METURAY_LINUX)
        return 64 - __builtin_clzl(val);
    #endif
}

uint32_t Utility::FindLastSet32(uint32_t val)
{
    #ifdef METURAY_WIN
        unsigned long ul;
        _BitScanReverse(&ul, val);
        return ul;
    #elif defined(METURAY_LINUX)
        return 32 - __builtin_clz(val);
    #endif
}