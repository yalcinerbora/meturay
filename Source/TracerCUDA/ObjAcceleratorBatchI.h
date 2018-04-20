#pragma once

struct ConstRayStackGMem;

class ObjAcceleratorBatchI
{
	public:
		virtual			~ObjAcceleratorBatchI() = default;

		//
		virtual void	CheckObjectHits(float* gDistances,
										HitRecordGMem gHits,
										//
										const uint32_t* objectIds,
										const uint32_t* rayIndex,
										const ConstRayStackGMem gRays,
										const uint32_t currentRayCount) = 0;
};