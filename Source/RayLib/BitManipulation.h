#pragma once

#include <cstdint>
#include "Types.h"

// CUDA Does not have <bit> header available
// atm so we define these ton wrapper functions

namespace Utility
{
    template<class T>
    T       FindLastSet(T);

    template<class T>
    T       NextPowOfTwo(T);

    template<class T>
    int     BitCount(T);
}

extern template int Utility::BitCount<uint32_t>(uint32_t);
extern template int Utility::BitCount<uint64_t>(uint64_t);

extern template uint32_t Utility::NextPowOfTwo<uint32_t>(uint32_t);
extern template uint64_t Utility::NextPowOfTwo<uint64_t>(uint64_t);

extern template uint32_t Utility::FindLastSet<uint32_t>(uint32_t);
extern template uint64_t Utility::FindLastSet<uint64_t>(uint64_t);