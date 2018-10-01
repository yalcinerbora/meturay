#pragma once
/**


*/
#include <cstdint>

typedef uint32_t HitKey;
typedef uint32_t RayId;

namespace HitConstants
{
	static constexpr HitKey		InvalidKey = 0xFFFFFFFF;
	static constexpr HitKey		OutsideMatKey = 0xFFFFFFFE;
	static constexpr uint32_t	InvalidData = 0xFFFFFFFF;
}