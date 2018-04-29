#pragma once

#include <cstdint>

struct RayRecordGMem;
struct ConstHitRecordGMem;
struct ConstRayRecordGMem;

class GPUMaterialI
{
	public:
		virtual						~GPUMaterialI() = default;

		// Interface
		virtual void				BounceRays(RayRecordGMem& gOutRays,
											   const ConstHitRecordGMem& gHits,
											   const ConstRayRecordGMem& gRays,
											   size_t rayCount) = 0;

};
