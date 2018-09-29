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
		virtual					~GPUMaterialI() = default;

		// Interface
		virtual void			ShadeRays(RayGMem* dRayOut,
										  void* dRayAuxOut,
										  //  Input
										  const RayGMem* dRayIn,
										  const HitKey* dCurrentHits,
										  const void* dRayAuxIn,
										  const RayId* dRayIds,

										  const uint32_t rayCount,
										  RNGMemory& rngMem) = 0;

		virtual uint8_t			MaxOutRayPerRay() const = 0;
};
