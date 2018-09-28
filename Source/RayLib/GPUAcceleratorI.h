#pragma once
/**

Base Interface for GPU accelerators

*/

#include <cstdint>
#include "HitStructs.h"

struct RayGMem;

class GPUAcceleratorGroupI
{
	public:
		virtual					~GPUAcceleratorGroupI() = default;

		// Interface
		// Kernel Logic: For each ray (but dRays & dHit
		virtual void			Hit(// Output
									HitGMem* dHits,
									// Inputs
									const RayGMem* dRays,
									const RayId* dRayIds,
									uint32_t rayCount) = 0;

};

class GPUBaseAcceleratorI
{
	public:
		virtual					~GPUBaseAcceleratorI() = default;

		// Interface
		// Base accelerator only points to the next accelerator key.
		// It can return invalid key,
		// which is either means data is out of bounds or ray is invalid.
		virtual void			Hit(// Output
									HitKey* dKeys,
									// Inputs
									const RayGMem* dRays,									
									const RayId* dRayIds,
									uint32_t rayCount) = 0;
};