#pragma once

#include "HitStructs.h"
#include <string>

namespace Debug
{
	void PrintHitPairs(const RayId* ids, const HitKey* keys, size_t count);
	void WriteHitPairs(const RayId* ids, const HitKey* keys, size_t count, const std::string& file);

	void PrintHitGMem(const HitGMem* hits, size_t count);
	void WriteHitGMem(const HitGMem* hits, size_t count, const std::string& fileName);

	void PrintRayIds(const RayId* ids, size_t count);
	void WriteRayIds(const RayId* ids, size_t count, const std::string& fileName);
}