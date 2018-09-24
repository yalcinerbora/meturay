#pragma once
/**


*/
#include <cstdint>

typedef uint32_t HitKey;
typedef uint32_t RayId;

struct alignas(8) HitGMem
{
	uint32_t	hitKey;			// Represents which accelerator holds its hit data
	uint32_t	innerId;		// Inner index of that accelerator
};