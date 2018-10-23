#pragma once

#include "GPUMaterialI.h"
#include "MaterialKernels.cuh"

// Partial Implementations
template <class TLogic, class MaterialData, class Surface,
	ShadeFunc<TLogic, Surface, MaterialData> ShadeFunction>
class GPUMaterialGroup : public GPUMaterialGroupI
{
	public:
		// Types from 
		using MaterialData				= typename MaterialData;
		using Surface					= typename Surface;

		static const auto ShadeFunc		= ShadeFunction;

	private:

	protected:
		MaterialData					matData = MaterialData{};

	public:
		// Constructors & Destructor
										GPUMaterialGroup() = default;
		virtual							~GPUMaterialGroup() = default;
};

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceFunc>
class GPUMaterialBatch : public GPUMaterialBatchI
{
	public:
		static constexpr auto SurfFunc		= SurfaceFunc;

	private:
		const MGroup&						materialGroup;
		const PGroup&						primitiveGroup;

	protected:
	public:
		// Constrcutors & Destructor
											GPUMaterialBatch(const GPUMaterialGroupI& m,
															 const GPUPrimitiveGroupI& p);
											~GPUMaterialBatch() = default;

		// Interface
		// KC
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
													  //
													  const uint32_t rayCount,
													  RNGMemory& rngMem) const override;

	
		const GPUPrimitiveGroupI&			PrimitiveGroup() const override { return pGroup; }
		const GPUMaterialGroupI&			MaterialGroup() const override { return mGroup; }
};

template <class T, class M, class P, SurfaceFunc<M, P> SF>
GPUMaterialBatch<T, M, P, SF>::GPUMaterialBatch(const GPUMaterialGroupI& m,
												const GPUPrimitiveGroupI& p)
	: materialGroup(static_cast<const M&>(m))
	, primitiveGroup(static_cast<const P&>(p))
{}

template <class T, class M, class P, SurfaceFunc<M, P> SF>
void GPUMaterialBatch<T, M, P, SF>::ShadeRays(// Output
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
	using PrimitiveData = typename P::PrimitiveData;
	using MaterialData = typename M::MaterialData;
	using RayAuxiliary = typename T::RayAuxiliary;
	
	// TODO: Is there a better way to implement this
	PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
	MaterialData matData = MatDataAccessor::Data(materialGroup);

	// Test
	KCMaterialShade<T, M, P, decltype(*this)>>//<<<1, 1>>>
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

template <class T, class M, class P, SurfaceFunc<M, P> SF>
const GPUPrimitiveGroupI& GPUMaterialBatch<T, M, P, SF>::PrimitiveGroup() const
{ 
	return pGroup;
}

template <class T, class M, class P, SurfaceFunc<M, P> SF>
const GPUMaterialGroupI& GPUMaterialBatch<T, M, P, SF>::MaterialGroup() const
{
	return mGroup;
}