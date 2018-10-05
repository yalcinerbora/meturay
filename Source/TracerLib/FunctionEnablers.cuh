/**

*/

#include <type_traits>
#include <vector_types.h>

#include "RayLib/Vector.h"
#include "RayLib/Matrix.h"
#include "RayLib/Quaternion.h"

#define ENABLE_NV_TYPE(count) \
		std::is_same<T, float##count>::value		|| \
		std::is_same<T, double##count>::value		|| \
		std::is_same<T, short##count>::value		|| \
		std::is_same<T, ushort##count>::value		|| \
		std::is_same<T, int##count>::value			|| \
		std::is_same<T, uint##count>::value			|| \
		std::is_same<T, long##count>::value			|| \
		std::is_same<T, ulong##count>::value		|| \
		std::is_same<T, long##count>::value			|| \
		std::is_same<T, ulong##count>::value

// Enable Ifs
template<class T, class O>
using EnableNV1Type = typename std::enable_if<ENABLE_NV_TYPE(1), O>::type;

template<class T, class O>
using EnableNV2Type = typename std::enable_if<ENABLE_NV_TYPE(2), O>::type;

template<class T, class O>
using EnableNV3Type = typename std::enable_if<ENABLE_NV_TYPE(3), O>::type;

template<class T, class O>
using EnableNV4Type = typename std::enable_if<ENABLE_NV_TYPE(4), O>::type;

template<class T, class O>
using EnableArithmetic = typename std::enable_if<std::is_arithmetic<T>::value, O>::type;

template<class T, class O>
using EnableVectorOrMatrix = typename std::enable_if<IsVectorType<T>::value ||
													 IsMatrixType<T>::value, O>::type;

template<class T, class O>
using EnableRest = typename std::enable_if<IsVectorType<T>::value ||
										   IsMatrixType<T>::value ||
										   IsQuatType<T>::value ||
										   std::is_arithmetic<T>::value, O>::type;