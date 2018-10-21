#pragma once

#include "GPUMaterialI.h"
#include "MaterialKernels.cuh"

// Template
template <class TLogic, class PGroup, class MGroup>
class GPUMaterialBatch final : public GPUMaterialBatchI
{
	public:
		MGroup&								materialGroup;
		PGroup&								primitiveGroup;

	private:
	protected:
	public:
		// Constrcutors & Destructor
											GPUMaterialBatch(MGroup& m, PGroup& p) : mGroup(m), pGroup(p) {}
											~GPUMaterialBatch() = default;

		// Interface
		void								ShadeRays(// Output
													  RayGMem* dRayOut,
													  void* dRayAuxOut,
													  //  Input
													  const RayGMem* dRayIn,
													  const void* dRayAuxIn,
													  const PrimitiveId* dPrimitiveIds,
													  const HitStructPtr dHitStructs,
													  //
													  const HitKey* dMatIds,
													  const RayId* dRayIds,

													  const uint32_t rayCount,
													  RNGMemory& rngMem) const override;

	
		const GPUPrimitiveGroupI&			PrimitiveGroup() const override { return pGroup; }
		const GPUMaterialGroupI&			MaterialGroup() const override { return mGroup; }


		static __device__ MGroup::Surface	SurfFunc(const PGroup::HitReg&,
													 const PGroup::PData&);
};

template <class T, class P, class M>
void GPUMaterialBatch<T, P, M>::ShadeRays(// Output
										  RayGMem* dRayOut,
										  void* dRayAuxOut,
										  //  Input
										  const RayGMem* dRayIn,
										  const void* dRayAuxIn,
										  const PrimitiveId* dPrimitiveIds,
										  const HitStructPtr dHitStructs,
										  //
										  const HitKey* dMatIds,
										  const RayId* dRayIds,

										  const uint32_t rayCount,
										  RNGMemory& rngMem) const
{	
	// TODO: Is there a better way to implement this
	using PrimitiveData = typename P::PrimitiveData;
	using MaterialData = typename M::MaterialData;
	using RayAuxiliary = typename T::RayAuxiliary;

	PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
	MaterialData matData = MaterialDataAccessor::Data(materialGroup);

	// Test
	KCMaterialShade<T, P, M>//<<<1, 1>>>
	(
		// Output
		dRayOut,
		static_cast<RayAuxiliary*>(dRayAuxOut),
		materialGroup.MaxOutRay(),
		// Input
		dRayIn,
		static_cast<RayAuxiliary*>(dRayAuxIn),
		dPrimitiveIds,
		dHitStructs,
		//
		dMatIds,
		dRayIds,
		//
		rayCount,		
		rngMem,
		// Material Related
		matData,
		// Primitive Related
		primData
	);
}