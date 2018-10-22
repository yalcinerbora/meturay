#pragma once

#include "GPUMaterialI.h"
#include "MaterialKernels.cuh"

//template <class Surface, class RayAuxiliary, class MaterialData>

template <class TLogic, class Surface, class MaterialData>
class GPUMaterialGroup : public GPUMaterialGroupI
{
	public:
		using RayAux						= typename TLogic::RayAux;
		using MaterialData					= typename MaterialData;
		using Surface						= typename Surface;

		static const auto ShadeFunc			= ShadeFunction;

	private:
		// Shade Function Signature
		//
		__device__ __host__
		static void							ShadeFunction(// Output
														  RayGMem* gOutRays,
														  RayAuxiliary* gOutRayAux,
														  const uint32_t maxOutRay,
														  // Input as registers
														  const RayReg& ray,
														  const Surface& surface,
														  const RayAuxiliary& aux,
														  // 
														  RandomGPU& rng,
														  // Input as global memory
														  const MaterialData& gMatData,
														  const HitKey::Type& matId);

	protected:
		MaterialData						matData;

	public:
		// Constructor & Destructor
											GPUMaterialGroup(MaterialData) = default;
};


template <class TLogic, class MGroup, class PGroup>
class GPUMaterialBatch final  : public GPUMaterialBatchI
{
	public:
	static constexpr auto SurfFunc			= SurfaceFunc<MGroup, PGroup>;

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
		// Type (as string) of the primitive group
		const char*							Type() const override;
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

template <class T, class M, class P>
GPUMaterialBatch<T, M, P>::GPUMaterialBatch(const GPUMaterialGroupI& m,
											const GPUPrimitiveGroupI& p)
	: materialGroup(static_cast<const M&>(m))
	, primitiveGroup(static_cast<const P&>(p))
{}

template <class T, class M, class P>
void GPUMaterialBatch<T, M, P>::ShadeRays(// Output
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
	KCMaterialShade<T, P, M, GPUMaterialBatch<T, M, P>><<<1, 1>>>
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

template <class T, class M, class P>
const GPUPrimitiveGroupI& GPUMaterialBatch<T, M, P>::PrimitiveGroup() const
{ 
	return pGroup;
}

template <class T, class M, class P>
const GPUMaterialGroupI& GPUMaterialBatch<T, M, P>::MaterialGroup() const
{
	return mGroup;
}