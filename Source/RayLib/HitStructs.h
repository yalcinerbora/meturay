#pragma once
/**


*/
#include <cstdint>

typedef uint32_t HitKey;

struct alignas(8) HitId
{
	uint32_t rayId;
	uint32_t innerId;
};