#pragma once
/**

Sub-space of the array with a unique id

*/

template<class T>
struct ArrayPortion
{
    T           portionId;
    size_t      offset;
    size_t      count;

    bool        operator<(const ArrayPortion&) const;
    bool        operator<(uint32_t portionId) const;
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