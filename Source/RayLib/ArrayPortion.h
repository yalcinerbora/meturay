#pragma once
/**

Sub-space of the array with a unique id

*/

#include <vector>
#include <cstdint>

template<class T>
struct ArrayPortion
{
    T           portionId;
    size_t      offset;
    size_t      count;

    bool        operator<(const ArrayPortion&) const;
    bool        operator<(uint32_t portionId) const;
};

template <class T>
struct MultiArrayPortion
{
    T                       portionId;
    std::vector<size_t>     offsets;
    std::vector<size_t>     counts;

    bool                    operator<(const MultiArrayPortion&) const;
    bool                    operator<(uint32_t portionId) const;
};


template<class T>
inline bool ArrayPortion<T>::operator<(const ArrayPortion& o) const
{
    return portionId < o.portionId;
}

template<class T>
inline bool ArrayPortion<T>::operator<(uint32_t pId) const
{
    return portionId < pId;
}

template<class T>
inline bool MultiArrayPortion<T>::operator<(const MultiArrayPortion& o) const
{
    return portionId < o.portionId;
}

template<class T>
inline bool MultiArrayPortion<T>::operator<(uint32_t pId) const
{
    return portionId < pId;
}