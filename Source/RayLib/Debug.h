#pragma once

#include "HitStructs.h"
#include <string>

namespace Debug
{
	void PrintHitPairs(const RayId* ids, const HitKey* keys, size_t count);
	void WriteHitPairs(const RayId* ids, const HitKey* keys, size_t count, const std::string& file);
}