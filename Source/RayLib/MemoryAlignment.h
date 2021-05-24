#pragma once

namespace Memory
{
    static constexpr const size_t AlignByteCount = 16;

    constexpr size_t AlignSize(size_t size);
    constexpr size_t AlignSize(size_t size, size_t alignCount);
}

inline constexpr
size_t Memory::AlignSize(size_t size)
{
    return AlignSize(size, AlignByteCount);
}

inline constexpr
size_t Memory::AlignSize(size_t size, size_t alignCount)
{
    size = alignCount * ((size + alignCount - 1) / alignCount);
    return size;
}