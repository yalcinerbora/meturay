#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

template<class T>
using ComparisonFunc = bool(&)(const T&, const T&);

namespace GPUFunctions
{
    template<class T>
    __device__
    inline bool BinarySearchInBetween(float& index, T value, const T* list, int32_t size)
    {
        if(size == 0) return false;

        int32_t start = 0;
        int32_t end = size;
        while(start <= end)
        {
            int32_t mid = (start + end) / 2;

            T current = list[mid];
            T next = list[mid + 1];

            if(current <= value &&
               next > value)
            {
                T totalDist = next - current;
                T dist = value - current;
                index = static_cast<float>(mid) + (dist / totalDist);
                return true;
            }
            else if(value < current)
                end = mid - 1;
            else
                start = mid + 1;
        }
        return false;
    }
}