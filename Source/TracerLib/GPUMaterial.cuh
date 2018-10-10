#pragma once

#include "GPUMaterialI.h"
#include "MaterialKernels.cuh"

// Template
template <class TLogic, class PGroup, class MGroup>
class GPUMaterialBatch final : public GPUMaterialBatchI
{
	public:
		MGroup&								mGroup;
		PGroup&								pGroup;

	private:
	protected:
	public:
		// Constrcutors & Destructor
											GPUMaterialBatch(MGroup& m, PGroup& p) : mGroup(m), pGroup(p) {}
											~GPUMaterialBatch() = default;

		// Interface
		void								ShadeRays(RayGMem* dRayOut,
													  void* dRayAuxOut,
													  //  Input
													  const RayGMem* dRayIn,
													  const void* dHitStructs,
													  const void* dRayAuxIn,
													  const RayId* dRayIds,

													  const uint32_t rayCount,
													  RNGMemory& rngMem) const override;

	
		const GPUPrimitiveGroupI&			PrimitiveGroup() const override { return pGroup; }
		const GPUMaterialGroupI&			MaterialGroup() const override { return mGroup; }


		static __device__ MGroup::Surface	SurfFunc(const PGroup::HitReg&,
													 const PGroup::PData&);
};

template <class TLogic, class PGroup, class MGroup>
void GPUMaterialBatch<TLogic, PGroup, MGroup>::ShadeRays(RayGMem* dRayOut,
														 void* dRayAuxOut,
														 //  Input
														 const RayGMem* dRayIn,
														 const void* dHitStructs,
														 const void* dRayAuxIn,
														 const RayId* dRayIds,

														 const uint32_t rayCount,
														 RNGMemory& rngMem) const
{	
	// TODO: Delegate properly using gpuIds etc.


	// Test
	KCMaterialShade<TLogic,  PGroup, MGroup><<<1, 1>>>
	(
		dRayOut,
		dRayAuxOut,
		// RayRelated
		dRayIn,
		dHitStructs,
		dRayAuxIn,
		dRayIds,
		rayCount,
		// RNG Related
		rngMem,
		// Material Related
		mGroup.Data(),
		// Primitive Related
		pGroup.Data()
	);
}