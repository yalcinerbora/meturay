#pragma once

#include "GPUMaterialI.h"
#include "MaterialKernels.cuh"

// Partial Implementations
template <class TLogic, class MaterialD, class SurfaceD,
		  ShadeFunc<TLogic, SurfaceD, MaterialD> ShadeF>
class GPUMaterialGroup : public GPUMaterialGroupI
{
	public:
		// Types from 
		using MaterialData				= typename MaterialD;
		using Surface					= typename SurfaceD;

		static const auto ShadeFunc		= ShadeF;

	private:

	protected:
		MaterialData					matData = MaterialData{};

	public:
		// Constructors & Destructor
										GPUMaterialGroup() = default;
		virtual							~GPUMaterialGroup() = default;
};

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
class GPUMaterialBatch : public GPUMaterialBatchI
{
	public:
		//static constexpr auto SurfFunc		= SurfaceF;

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
	
		const GPUPrimitiveGroupI&			PrimitiveGroup() const override;
		const GPUMaterialGroupI&			MaterialGroup() const override;
};

//template <class T, class M, class P, SurfaceFunc<M, P> SF>
template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::GPUMaterialBatch(const GPUMaterialGroupI& m,
												const GPUPrimitiveGroupI& p)
	: materialGroup(static_cast<const MGroup&>(m))
	, primitiveGroup(static_cast<const PGroup&>(p))
{}

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
void GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::ShadeRays(// Output
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
	using PrimitiveData = typename PGroup::PrimitiveData;
	using MaterialData = typename MGroup::MaterialData;
	using RayAuxiliary = typename TLogic::RayAuxiliary;
	
	// TODO: Is there a better way to implement this
	PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
	MaterialData matData = MatDataAccessor::Data(materialGroup);

	// Test
	KCMaterialShade<TLogic, MGroup, PGroup, 
					GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>><<<1, 1>>>
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

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
const GPUPrimitiveGroupI& GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::PrimitiveGroup() const
{ 
	return pGroup;
}

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
const GPUMaterialGroupI& GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::MaterialGroup() const
{
	return mGroup;
}