#pragma once

#include <cstdint>

class GPUMaterialI
{
	public:
		virtual						~GPUMaterialI() = default;

		virtual void uint32_t		LocatedGPU() const = 0;
		virtual void				BounceRays(RayStackGMem& gOutRays,
											   const ConstHitRecordGMem gHits,
											   const ConstRayStackGMem& gRays) = 0;
};
