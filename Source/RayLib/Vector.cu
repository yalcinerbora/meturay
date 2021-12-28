#include "Vector.h"

// Vector Define
// Float Type
// template class Vector<2, float>;
// template class Vector<3, float>;
// template class Vector<4, float>;
// Double Type
// template class Vector<2, double>;
// template class Vector<3, double>;
// template class Vector<4, double>;
// Integer Type
// template class Vector<2, int32_t>;
// template class Vector<3, int32_t>;
// template class Vector<4, int32_t>;
// Unsigned Integer Type
// IMPORTANT: This (and the non extern brother in the Vector.cu)
// Breaks the at the __cudaRegisterFatbinary() when you load VisorGL.so.
// I don't know why...
// template class Vector<2, uint32_t>;
// template class Vector<3, uint32_t>;
// template class Vector<4, uint32_t>;
// Long Types
template class Vector<2, int64_t>;
template class Vector<2, uint64_t>;
// Short Type
template class Vector<2, int16_t>;
template class Vector<3, int16_t>;
template class Vector<4, int16_t>;
// Unsigned Short Type
template class Vector<2, uint16_t>;
template class Vector<3, uint16_t>;
template class Vector<4, uint16_t>;
// Byte Type
template class Vector<2, int8_t>;
template class Vector<3, int8_t>;
template class Vector<4, int8_t>;
// Unsigned Byte Type
template class Vector<2, uint8_t>;
template class Vector<3, uint8_t>;
template class Vector<4, uint8_t>;