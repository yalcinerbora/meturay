#pragma once

#include "HitStructs.cuh"
#include "RayStructs.h"
#include "Random.cuh"

template <class MGroup, class PGroup>
using SurfaceFunc = MGroup::Surface(&)(const PGroup::PrimitiveData&,
									   const PGroup::HitData&,
									   PrimitiveId);
//
template <class TLogic, class Surface, class MaterialData>
using ShadeFunc = void(*)(// Output
						  RayGMem* gOutRays,
						  TLogic::RayAuxiliary* gOutRayAux,
						  const uint32_t maxOutRay,
						  // Input as registers
						  const RayReg& ray,
						  const Surface& surface,
						  const TLogic::RayAuxiliary& aux,
						  // 
						  RandomGPU& rng,
						  // Input as global memory
						  const MaterialData& gMatData,
						  const HitKey::Type& matId);

template <class TLogic, class MGroup, class PGroup, class BGroup>
__global__ void KCMaterialShade(// Output
								RayGMem* gOutRays,
								TLogic::RayAuxiliary* gOutRayAux,
								const uint32_t maxOutRay,
								// Input								
								const RayGMem* gInRays,
								const TLogic::RayAuxiliary* gInRayAux,
								const PrimitiveId* gPrimitiveIds,
								const HitStructPtr gHitStructs,
								//
								const HitKey* gMatIds,
								const RayId* gRayIds,
								//
								const uint32_t rayCount,								
								RNGGMem gRNGStates,
								// Material Related
								const MGroup::MaterialData matData,
								// Primitive Related
								const PGroup::PrimitiveData primData)
{
	// Fetch Types from Template Classes
	using HitData = typename PGroup::HitData;				// HitData is defined by primitive
	using Surface = typename MGroup::Surface;				// Surface is defined by material group
	using RayAuxiliary = typename TLogic::RayAuxiliary;		// Hit register defined by primitive

	// Pre-grid stride loop
	// RNG is allocated for each SM (not for each thread)
	extern __shared__ uint32_t sRNGStates[];
	RandomGPU rng(gRNGStates.state, sRNGStates);

	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; globalId += blockDim.x * gridDim.x)
	{
		const RayId rayId = gRayIds[globalId];
		const HitKey hitKey = gMatIds[globalId];

		// Load Input to Registers
		const RayReg ray(gInRays, rayId);
		const RayAuxiliary aux = gInRayAux[rayId];
		const PrimitiveId gPrimitiveId = gPrimitiveIds[rayId];

		// Generate surface data from hit
		const HitData hit = gHitStructs[rayId];
		const Surface surface = BGroup::SurfFunc(hit, primData,
												 gPrimitiveId);

		// Determine Output Location
		RayGMem* gLocalRayOut = gOutRays + globalId * maxOutRay;
		RayAuxiliary* gLocalAuxOut = gOutRayAux + globalId * maxOutRay;
		// Actual Shading
		MGroup::ShadeFunc(// Output
						  gLocalRayOut,
						  gLocalAuxOut,
						  maxOutRay,
						  // Input as registers
						  ray,
						  surface,
						  aux,
						  //
						  rng,
						  // Input as global memory
						  matData,
						  HitKey::FetchIdPortion(hitKey));
	}
}