#pragma once
/**

Base Interface for GPU accelerators

*/

#include <cstdint>
#include "HitStructs.h"

struct RayGMem;

class GPUAcceleratorI
{
	public:
		virtual			~GPUAcceleratorI() = default;

		// Interface
		virtual void			Hit(HitId* dHitIds,
									const HitKey* dKeys,
									const RayGMem* dRays,									
									uint32_t rayCount, 
									bool traversePartial = false) = 0;

};