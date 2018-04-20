#pragma once

#include <cstdint>

struct ConstRayStackGMem;

class SceneAcceleratorI
{
	public:
		virtual				~SceneAcceleratorI() = default;

		//
		virtual void		FindNextObjectToCheck(uint32_t* objectId,
												  uint32_t* loopIndex,
												  uint32_t* rayIndexOut,

												  uint32_t& allocator,

												  const float* objectHits,
												  const uint32_t* rayIndexIn,
												  const ConstRayStackGMem gRays,
												  const uint32_t currentRayCount) = 0;

};
