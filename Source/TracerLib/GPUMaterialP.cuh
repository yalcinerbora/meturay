#pragma once

#include "GPUMaterialI.h"
#include "MaterialKernels.cuh"
#include "GPUPrimitiveP.cuh"
#include "RNGMemory.h"

#include "RayLib/CompileTimeString.h"

struct MatDataAccessor;

template <class MaterialD>
class GPUMaterialGroupP
{
	friend struct MatDataAccessor;

	protected:
		MaterialD dData = MaterialD{};
};

// Partial Implementations
template <class TLogic, class MaterialD, class SurfaceD,
		  ShadeFunc<TLogic, SurfaceD, MaterialD> ShadeF>
class GPUMaterialGroup 
	: public GPUMaterialGroupI
	, public GPUMaterialGroupP<MaterialD>
{
	public:
		// Types from 
		using MaterialData				= typename MaterialD;
		using Surface					= typename SurfaceD;

		static constexpr auto ShadeFunc	= ShadeF;

	private:
	protected:
	
	public:
		// Constructors & Destructor
										GPUMaterialGroup() = default;
		virtual							~GPUMaterialGroup() = default;
};

#include <array>

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
class GPUMaterialBatch final : public GPUMaterialBatchI
{
	public:
		static constexpr auto SurfFunc		= SurfaceF;
		static const char*					TypeName;

	private:
		const MGroup&						materialGroup;
		const PGroup&						primitiveGroup;

		static const std::string			TypeNamePriv;


	protected:		
	public:
		// Constrcutors & Destructor
											GPUMaterialBatch(const GPUMaterialGroupI& m,
															 const GPUPrimitiveGroupI& p);
											~GPUMaterialBatch() = default;
											
		// Type (as string) of the primitive group
		const char*							Type() const override;
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

		uint8_t								OutRayCount() const override;
};

struct MatDataAccessor
{
	// Data fetch function of the primitive
	// This struct should contain all necessary data required for kernel calls
	// related to this primitive
	// I dont know any design pattern for converting from static polymorphism
	// to dynamic one. This is my solution (it is quite werid)
	template <class MaterialGroupS>
	static typename MaterialGroupS::MaterialData Data(const MaterialGroupS& mg)
	{
		using M = typename MaterialGroupS::MaterialData;
		return static_cast<const GPUMaterialGroupP<M>&>(mg).dData;
	}
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
const char* GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::TypeName = TypeNamePriv.c_str();

template <class TLogic, class MGroup, class PGroup,
	SurfaceFunc<MGroup, PGroup> SurfaceF>
	const std::string GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::TypeNamePriv = std::string(MGroup::TypeName) + PGroup::TypeName;

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
const char* GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::Type() const
{
	return TypeName;
}

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
	using RayAuxData = typename TLogic::RayAuxData;
	
	// TODO: Is there a better way to implement this
	const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
	const MaterialData matData = MatDataAccessor::Data(materialGroup);

	const uint32_t outRayCount = materialGroup.OutRayCount();

	// Test
	KCMaterialShade<TLogic, MGroup, PGroup, SurfFunc><<<1,1>>>
	(
		// Output
		dRayOut,
		static_cast<RayAuxData*>(dRayAuxOut),
		outRayCount,
		// Input
		dRayIn,
		static_cast<const RayAuxData*>(dRayAuxIn),
		dPrimitiveIds,
		dHitStructs,
		//
		dMatIds,
		dRayIds,
		//
		rayCount,		
		rngMem.RNGData(0),
		// Material Related
		matData,
		// Primitive Related
		primData
	);
	CUDA_KERNEL_CHECK();
}

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
const GPUPrimitiveGroupI& GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::PrimitiveGroup() const
{ 
	return primitiveGroup;
}

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
const GPUMaterialGroupI& GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::MaterialGroup() const
{
	return materialGroup;
}

template <class TLogic, class MGroup, class PGroup,
	SurfaceFunc<MGroup, PGroup> SurfaceF>
uint8_t GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::OutRayCount() const
{
	return materialGroup.OutRayCount();
}