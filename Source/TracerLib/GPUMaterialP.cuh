#pragma once

#include <array>

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

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
class GPUMaterialBatch final : public GPUMaterialBatchI
{
	private:
		static const std::string			TypeNamePriv;

	public:
		static constexpr auto SurfFunc		= SurfaceF;
		static const char*					TypeName;

	private:
		const MGroup&						materialGroup;
		const PGroup&						primitiveGroup;
		const int							gpuId;

	protected:		
	public:
		// Constrcutors & Destructor
											GPUMaterialBatch(const GPUMaterialGroupI& m,
															 const GPUPrimitiveGroupI& p,
															 int gpuId);
											~GPUMaterialBatch() = default;
											
		// Type (as string) of the primitive group
		const char*							Type() const override;
		// Interface
		// KC
		void								ShadeRays(// Output
													  Vector4* dPixels,
													  //
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
		int									GPUId() const override;

		uint8_t								OutRayCount() const override;
};

template <class TLogic, class MaterialD,
		  BoundaryShadeFunc<TLogic, MaterialD> ShadeF>
class GPUBoundaryMatGroup	
	: public GPUMaterialGroupI
	, public GPUMaterialGroupP<MaterialD>
{	
	public:
		// Types from 
		using MaterialData				= typename MaterialD;

		static constexpr auto ShadeFunc = ShadeF;

		private:
		protected:

		public:
			// Constructors & Destructor
										GPUBoundaryMatGroup() = default;
			virtual						~GPUBoundaryMatGroup() = default;
};

template <class TLogic, class MGroup>
class GPUBoundaryMatBatch final : public GPUMaterialBatchI
{
	private:
		static const std::string			TypeNamePriv;

	public:
		static const char*					TypeName;

	private:
		const MGroup&						materialGroup;
		static const GPUPrimitiveGroupI*	primitiveGroup;
		const int							gpuId;

	protected:		
	public:
		// Constrcutors & Destructor
											GPUBoundaryMatBatch(const GPUMaterialGroupI& m,
															    const GPUPrimitiveGroupI& p,
																int gpuId);
											~GPUBoundaryMatBatch() = default;
											
		// Type (as string) of the primitive group
		const char*							Type() const override;
		// Interface
		// KC
		void								ShadeRays(// Output
													  Vector4* dPixels,
													  //
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
		int									GPUId() const override;

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

template <class TLogic, class MGroup, class PGroup,
	SurfaceFunc<MGroup, PGroup> SurfaceF>
	GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::GPUMaterialBatch(const GPUMaterialGroupI& m,
																		 const GPUPrimitiveGroupI& p,
																		 int gpuId)
	: materialGroup(static_cast<const MGroup&>(m))
	, primitiveGroup(static_cast<const PGroup&>(p))
	, gpuId(gpuId)
{}

template <class TLogic, class MGroup, class PGroup,
	SurfaceFunc<MGroup, PGroup> SurfaceF>
	const std::string GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::TypeNamePriv = std::string(MGroup::TypeName) + PGroup::TypeName;

template <class TLogic, class MGroup, class PGroup,
	SurfaceFunc<MGroup, PGroup> SurfaceF>
	const char* GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::TypeName = TypeNamePriv.c_str();

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
const char* GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::Type() const
{
	return TypeName;
}

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
void GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::ShadeRays(// Output
																   Vector4* dPixels,
																   //
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
		dPixels,
		//
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
		rngMem.RNGData(gpuId),
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
int GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::GPUId() const
{
	return gpuId;
}

template <class TLogic, class MGroup, class PGroup,
		  SurfaceFunc<MGroup, PGroup> SurfaceF>
uint8_t GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::OutRayCount() const
{
	return materialGroup.OutRayCount();
}

template <class TLogic, class MGroup>
GPUBoundaryMatBatch<TLogic, MGroup>::GPUBoundaryMatBatch(const GPUMaterialGroupI& m,
														 const GPUPrimitiveGroupI& p,
														 int gpuId)
	: materialGroup(static_cast<const MGroup&>(m))
	, gpuId(gpuId)
{}

template <class TLogic, class MGroup>
const std::string GPUBoundaryMatBatch<TLogic, MGroup>::TypeNamePriv = std::string(MGroup::TypeName);

template <class TLogic, class MGroup>
const GPUPrimitiveGroupI* GPUBoundaryMatBatch<TLogic, MGroup>::primitiveGroup = nullptr;

template <class TLogic, class MGroup>
const char* GPUBoundaryMatBatch<TLogic, MGroup>::Type() const
{
	return TypeNamePriv.c_str();
}

template <class TLogic, class MGroup>
void GPUBoundaryMatBatch<TLogic, MGroup>::ShadeRays(// Output
													Vector4* dPixels,
													//
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
													RNGMemory& rngMem) const
{
	using MaterialData = typename MGroup::MaterialData;
	using RayAuxData = typename TLogic::RayAuxData;
	
	// TODO: Is there a better way to implement this
	const MaterialData matData = MatDataAccessor::Data(materialGroup);	

	// Test
	KCBoundaryMatShade<TLogic, MGroup><<<1,1>>>
	(
		// Output
		dPixels,
		// Input
		dRayIn,
		static_cast<const RayAuxData*>(dRayAuxIn),		
		//
		dRayIds,
		//
		rayCount,		
		rngMem.RNGData(gpuId),
		// Material Related
		matData
	);
	CUDA_KERNEL_CHECK();
}

template <class TLogic, class MGroup>
const GPUPrimitiveGroupI& GPUBoundaryMatBatch<TLogic, MGroup>::PrimitiveGroup() const
{
	return *primitiveGroup;
}

template <class TLogic, class MGroup>
const GPUMaterialGroupI& GPUBoundaryMatBatch<TLogic, MGroup>::MaterialGroup() const
{
	return materialGroup;
}

template <class TLogic, class MGroup>
int GPUBoundaryMatBatch<TLogic, MGroup>::GPUId() const
{
	return gpuId;
}

template <class TLogic, class MGroup>
uint8_t GPUBoundaryMatBatch<TLogic, MGroup>::OutRayCount() const
{
	return materialGroup.OutRayCount();
}