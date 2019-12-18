#include "BitManipulation.h"
#include "System.h"

uint64_t Utility::FindFirstSet(uint64_t val)
{
    #ifdef _WIN32
        unsigned long ul;
        _BitScanReverse64(&ul, val);
        return ul;
    #endif
}