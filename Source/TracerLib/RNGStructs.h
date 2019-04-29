#pragma once

#include <cstdint>

struct ConstRNGGMem
{
    const uint32_t* state;  
};

struct RNGGMem
{
    uint32_t* state;

    constexpr operator ConstRNGGMem() const;
};

constexpr RNGGMem::operator ConstRNGGMem() const
{
    return {state};
}
