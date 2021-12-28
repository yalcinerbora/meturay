#include "BitManipulation.h"
#include <bit>
#include "System.h"

template<class T>
int Utility::BitCount(T val)
{
    return std::popcount(val);
}

template<class T>
T Utility::NextPowOfTwo(T val)
{
    return std::bit_ceil(val);
}

template<class T>
T Utility::FindLastSet(T val)
{
    return (sizeof(T) * BYTE_BITS) - std::countl_zero(val) - 1;
}

template int Utility::BitCount<uint32_t>(uint32_t);
template int Utility::BitCount<uint64_t>(uint64_t);

template uint32_t Utility::NextPowOfTwo<uint32_t>(uint32_t);
template uint64_t Utility::NextPowOfTwo<uint64_t>(uint64_t);

template uint32_t Utility::FindLastSet<uint32_t>(uint32_t);
template uint64_t Utility::FindLastSet<uint64_t>(uint64_t);