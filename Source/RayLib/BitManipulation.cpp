#include "BitManipulation.h"
#include "System.h"

uint64_t Utility::FindLastSet64(uint64_t val)
{
    #ifdef _WIN32
        unsigned long ul;
        _BitScanReverse64(&ul, val);
        return ul;
    #endif
}

uint32_t Utility::FindLastSet32(uint32_t val)
{
    #ifdef _WIN32
        unsigned long ul;
        _BitScanReverse(&ul, val);
        return ul;
    #endif
}