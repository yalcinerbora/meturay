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
		virtual							~GPUAcceleratorGroupI() = default;

		// Interface
		// Kernel Logic
		virtual void					Hit(// I-O
											RayGMem* dRays,
											void* dHitStructs,
											HitKey* dCurrentHits,
											// Input
											const RayId* dRayIds,
											const HitKey* dPotentialHits,
											const uint32_t rayCount) const = 0;

		virtual const std::string&		AcceleratorType() const = 0;
		virtual const std::string&		PrimitiveType() const = 0;

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
									HitKey* dPotentialHits,
									// Inputs
									const RayGMem* dRays,									
									const RayId* dRayIds,
									const uint32_t rayCount) const = 0;
};