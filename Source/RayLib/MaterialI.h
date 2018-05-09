#pragma once

#include <cstdint>
/**

*/

struct RayRecordGMem;
struct ConstHitRecordGMem;
struct ConstRayRecordGMem;

class MaterialI
{
	public:
		virtual						~MaterialI() = default;

		// Interface
		virtual uint32_t			Id() const = 0;

		// Main Logic
		virtual void				BounceRays(// Outgoing Rays
											   RayRecordGMem& gOutRays,
											   // Incoming Rays
											   const ConstHitRecordGMem& gHits,
											   const ConstRayRecordGMem& gRays,
											   uint64_t rayCount) = 0;
};
