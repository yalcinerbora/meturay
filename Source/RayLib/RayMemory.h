#pragma once
/**

General Device memory manager for ray and it's auxiliary data


*/

#include "DeviceMemory.h"
#include "RayStructs.h"
#include "HitStructs.h"

#include <cstdint>
#include <cassert>

//template<class RayAuxData>
class RayMemory
{
	public:
		static constexpr size_t		AlignByteCount = 16;

	private:
		DeviceMemory				sortMemory;

		DeviceMemory				memHit;
		DeviceMemory				memIn;
		DeviceMemory				memOut;

		// Ray Related
		RayGMem*					dRayIn;
		RayGMem*					dRayOut;
		void*						dRayAuxIn;
		void*						dRayAuxOut;

		// Hit Related
		void*						dSortAuxiliary;
		HitId*						dIds0, *dIds1;		
		HitKey*						dKeys0, *dKeys1;
		HitId*						dIds;
		HitKey*						dKeys;
		
		static void					ResizeRayMemory(RayGMem*& dRays, void*& dRayAxData,
													DeviceMemory&, 
													size_t rayCount,
													size_t perRayAuxSize);

	public:
		// Constructors & Destructor
									RayMemory();
									RayMemory(const RayMemory&) = delete;
									RayMemory(RayMemory&&) = default;
		RayMemory&					operator=(const RayMemory&) = delete;
		RayMemory&					operator=(RayMemory&&) = default;
									~RayMemory() = default;

		// Accessors
		RayGMem*					Rays();
		const RayGMem*				Rays() const;
		HitId*						HitIds();
		const HitId*				HitIds() const;
		HitKey*						HitKeys();
		const HitKey*				HitKeys() const;
				
		// Memory Arrangement
		// Reset memory system (allocates for initial ray count)
		void						Reset(size_t rayCount);

		// Ray Related
		void						ResizeRayIn(size_t rayCount, size_t perRayAuxSize);
		void						ResizeRayOut(size_t rayCount, size_t perRayAuxSize);
		void						SwapRays(size_t rayCount);

		// Accelerator
		void						ResizeHitMemory(size_t rayCount);
		void						SortKeys(HitId*& ids, HitKey*& keys,
											 size_t count,
											 const Vector2i& bitRange);
};

inline RayGMem* RayMemory::Rays()
{
	return dRayStackIn;
}

inline const RayGMem* RayMemory::Rays() const
{
	return dRayStackIn;
}

inline HitId* RayMemory::HitIds()
{
	return dIds;
}

inline const HitId* RayMemory::HitIds() const
{
	return dIds;
}

inline HitKey* RayMemory::HitKeys()
{
	return dKeys;
}

inline const HitKey* RayMemory::HitKeys() const
{
	return dKeys;
}