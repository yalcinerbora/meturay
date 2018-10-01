#pragma once
/**


*/

#include <cstdint>
#include "HitStructs.h"

struct RayGMem;
class RNGMemory;

class GPUMaterialGroupI
{
	public:
		virtual					~GPUMaterialGroupI() = default;

		// Interface
		virtual void			ShadeRays(RayGMem* dRayOut,
										  void* dRayAuxOut,
										  //  Input
										  const RayGMem* dRayIn,
										  const void* dHitStructs,
										  const void* dRayAuxIn,
										  const RayId* dRayIds,

										  const uint32_t rayCount,
										  RNGMemory& rngMem) const = 0;

		virtual uint8_t			MaxOutRayPerRay() const = 0;
};


class GPUMaterialI
{
	public:
		
		// Load/Unload Material
		virtual void			LoadMaterial(int gpuId) = 0;
		virtual void			UnloadMaterial() = 0;
		// Query if it is loaded
		virtual bool			IsLoaded() = 0;


};