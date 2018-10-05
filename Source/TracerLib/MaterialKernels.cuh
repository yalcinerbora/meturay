#pragma once

#include "HitStructs.h"
#include "RayStructs.h"
#include "Random.cuh"

template <class Surface, class HitReg, class PrimitiveData>
using SurfaceFunc = __device__ Surface(*)(const HitReg& hit,
										  const PrimitiveData& gPrimData);

template <class HitReg>
using MatIdFetchFunc = __device__ MatId(*)(const HitReg&);

template <class Surface, class RayAuxiliary, class MaterialData>
using ShadeFunc = __device__ void(*)(// Output
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
									 const MatId& matId);

// This is fundemental Material traversal shader
// These kernels are generated by custom Surface Fetch and Shade Functions
// and Custom Primitives
template <class Surface, class RayAuxiliary,
		  class HitGMem, class HitReg,
		  class MaterialData, class PrimitiveData,
		  SurfaceFunc<Surface, HitReg, PrimitiveData> SurfFunc,
		  MatIdFetchFunc<HitReg> MatFunc,
		  ShadeFunc<Surface, PrimitiveData, MaterialData> ShadeFunc>
__global__ void KCMaterialShade(RayGMem* gOutRays,
								RayAuxiliary* gOutRayAux,
								const uint32_t maxOutRay,
								// Ray Related
								const RayGMem* gInRays,
								const HitGMem* gHitStructs,
								const RayAuxiliary* gInRayAux,
								const RayId* gRayIds,
								const uint32_t rayCount,
								// RNG Related
								RNGGMem gRNGStates,
								// Material Related
								const MaterialData gMatData,
								// Primitive Related
								const PrimitiveData gPrimData)
{
	// Pre-grid stride loop
	// RNG is allocated for each SM (not for each thread)
	__shared__ uint32_t sRNGStates[];
	RandomGPU rng(gRNGStates.state, sRNGStates);

	// Grid Stride Loop
	for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
		globalId < rayCount; i += blockDim.x * gridDim.x)
	{
		const RayId rayId = gRayIds[globalId];
		
		// Load Input to Registers
		const RayReg ray(gInRays, rayId);
		const RayAuxiliary aux = gInRayAux[rayId];

		// Generate surface data from hit
		const HitReg hit(gHitStructs, rayId);
		const Surface surface = SurfFunc(hit, gPrimData);
		const MatId matId = MatFunc(hit);

		// Determine Output Location
		RayGMem* gLocalRayOut = gOutRays + globalId * maxOutRay;
		RayAuxiliary gLocalAuxOut = gOutRayAux + globalId * maxOutRay;
		// Actual Shading
		ShadeFunc(// Output
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
				  gMatData,
				  matId);
	}
}