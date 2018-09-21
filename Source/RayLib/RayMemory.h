#pragma once
/**

General Device memory manager for ray and it's auxiliary data


*/

#include "RayStructs.h"
#include "HitStructs.h"

#include <cstdint>
#include <cassert>

template<class RayAuxData>
class RayMemory
{
	private:
		DeviceMemory				memRayIn;
		DeviceMemory				memRayOut;		
		DeviceMemory				memHit;

		// Ray Related
		RayGMem*					dRayStackIn;
		RayGMem*					dRayStackOut;
		RayAuxData*					dRayAuxIn;
		RayAuxData*					dRayAuxOut;

		// Hit Related
		HitId*						dHitRecord;
		HitKey*						dHitKeys;

		//static RayGMem				GenerateRayPtrs(void* mem, size_t rayCount);
		//static HitGMem				GenerateHitPtrs(void* mem, size_t rayCount);
		//static size_t				TotalMemoryForRay(size_t rayCount);
		//static size_t				TotalMemoryForHit(size_t rayCount);

	public:
		// Constructors & Destructor
									RayMemory();
									RayMemory(const RayMemory&) = delete;
									RayMemory(RayMemory&&) = default;
		RayMemory&					operator=(const RayMemory&) = delete;
		RayMemory&					operator=(RayMemory&&) = default;
									~RayMemory() = default;

		//// Accessors
		//RayGMem*					RayStackIn();
		//RayGMem*					RayStackOut();
		//HitGMem*					HitRecord();

		//const RayGMem*				RayStackIn() const;
		//const RayGMem*				RayStackOut() const;
		//const HitGMem*				HitRecord() const;
		
		// Memory Arrangement
		// Reset memory system (allocates for initial ray count)
		void						Reset(size_t rayCount);
		// Resize rayIn and Out
		void						ResizeRayIn(size_t rayCount);
		void						ResizeRayOut(size_t rayCount);
		void						ResizeHit(size_t rayCount);
		// Swap in and outs
		void						SwapRays(size_t rayCount);
};

// Implementation
#include "RayMemory.hpp"