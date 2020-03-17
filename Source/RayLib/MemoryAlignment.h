#pragma once

namespace Memory
{
    static constexpr const size_t AlignByteCount = 16;

    size_t AlignSize(size_t size);
    size_t AlignSize(size_t size, size_t alignCount);
}

inline size_t Memory::AlignSize(size_t size)
{
    return AlignSize(size, AlignByteCount);
}

inline size_t Memory::AlignSize(size_t size, size_t alignCount)
{
    size = alignCount * ((size + alignCount - 1) / alignCount);
    return size;
}