#include "Hitman.h"
#include "RayLib/RayMemory.h"
#include "RayLib/GPUAcceleratorI.h"

Hitman::Hitman(const HitmanOptions& opts)
	: opts(opts)
{}

void Hitman::Process(RayMemory& memory,
					 uint32_t rayCount)
{
	const RayGMem* dRays = memory.Rays();
	HitKey* dHitKeys = memory.HitKeys();
	HitId*	dHitIds = memory.HitIds();

	uint32_t currentRayCount = rayCount;
	while(currentRayCount > 0)
	{

		// Allocate auxiliary memory
		uint32_t* dInitialRayStates = nullptr;

		// Traverse accelerator
		baseAccelerator->Hit(dHitIds, dHitKeys,							 
							 dRays, currentRayCount, true);
		// Base accelerator traverses the data partially
		// It delegates the rays to smaller accelerators
		// by writing their Id's to its portion in the key.
		
		// After that systems sorts ray hit list and key
		// and partitions the array this partitioning scheme 

		// Sort initial results (to partition and launch kernels accordingly)
		// Parition to sub accelerators
		// reduce end count accordingly
		memory.SortKeys(dHitIds, dHitKeys, currentRayCount, opts.keyBitRange);
		auto portions = memory.Partition(currentRayCount, opts.keyBitRange);
	
		// For each partition
		for(const auto& p : portions)
		{
			// Skip
			if(p.portionId == RayMemory::InvalidKey) continue;

			auto loc = subAccelerators.find(p.portionId);
			if(loc == subAccelerators.end()) continue;

			// Parititon pointers
			const RayGMem* dRayStart = dRays + p.offset;
			const HitKey* dKeyStart = dHitKeys + p.offset;
			HitId* dIdStart = dHitIds + p.offset;

			loc->second->Hit(dHitIds, dHitKeys, dRays, 
							 static_cast<uint32_t>(p.count));

			// Those hits updated their internal hits etc
			// and updated values
		}
		// Iteration is done
	}
}