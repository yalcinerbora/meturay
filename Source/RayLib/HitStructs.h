#pragma once
/**


*/
#include <cstdint>

typedef uint32_t HitKey;

struct alignas(16) HitId
{
	uint32_t rayId;
	uint32_t innerId;
};