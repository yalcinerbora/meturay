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
};

template<class T>
inline bool ArrayPortion<T>::operator<(const ArrayPortion& o) const
{
    return portionId < o.portionId;
}