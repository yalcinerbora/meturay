#pragma once

#include <cstdint>
/**

*/

enum class SurfaceType;

struct RayRecordGMem;
struct ConstHitRecordGMem;
struct ConstRayRecordGMem;
struct Error;

#include "Vector.h"

class MaterialI
{
	public:
		virtual						~MaterialI() = default;

		// Interface
		virtual uint32_t			Id() const = 0;

		// Main Logic
		virtual void				BounceRays(// Outgoing Rays
											   RayRecordGMem gOutRays,
											   // Incoming Rays
											   const ConstHitRecordGMem gHits,
											   const ConstRayRecordGMem gRays,
											   // Limits
											   uint64_t rayCount,
											   // Surfaces
											   const Vector2ui* gSurfaceIndexList,
											   const void* gSurfaces,
											   SurfaceType) = 0;
		virtual Error				Load() = 0;
		virtual void				Unload() = 0;
};
