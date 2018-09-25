#pragma once
/**


*/

#include <cstdint>
#include "HitStructs.h"

struct RayGMem;
class RNGMemory;

class GPUMaterialI
{
	public:
		virtual					~GPUMaterialI() = 0;

		// Interface
		virtual void			ShadeRays(RayGMem* dRayOut,
										  void* dRayAuxOut,
										  //  Input
										  const RayGMem* dRayIn,
										  const HitGMem* dHitId,
										  const void* dRayAuxIn,
										  const RayId* dRayId,

										  const uint32_t rayCount,
										  RNGMemory& rngMem) = 0;

		virtual uint8_t			MaxOutRayPerRay() const = 0;
};
