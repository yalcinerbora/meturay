#pragma once
/**

Surface Interface

Surfaces are entities that can be hit upon.

*/

#include <cstdint>

struct HitRecordGMem;
struct ConstRayRecordGMem;

class SurfaceI
{
	public:
		virtual								~SurfaceI() = default;

		// Mesh unique identifier
		virtual uint32_t					Id() const = 0;
		virtual uint32_t					MaterialId() const = 0;

		// Main Logic
		virtual void						HitRays(const HitRecordGMem,
													const ConstRayRecordGMem,
													uint64_t rayCount) const = 0;
};