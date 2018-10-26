
#include "TracerLib/GPUPrimitiveSphere.h"
#include "TracerLib/GPUPrimitiveTriangle.h"


#include "TracerLib/GPUMaterialP.cuh"
#include "TracerLib/TracerLogicP.cuh"

//struct RayAuxBasic
//{
//	Vector3f totalIrradiance;
//};
//
//__device__ __host__
//void RayInitBasic(const RayAuxBasic*,
//				  const uint32_t writeLoc,
//				  // Input
//				  const RayAuxBasic defaults,
//				  // Index
//				  const Vector2ui& globalPixelId,
//				  const Vector2ui& localSampleId,
//				  const uint32_t samplePerPixel)
//{}
//
//template class TracerBaseLogic<RayAuxBasic, RayInitBasic>;
//
//struct StaticMaterialData
//{
//	Vector3 albedo;
//};
//
//struct EmptySurface
//{};
//
//__device__ __host__
////template <class TLogic, class Surface, class MaterialData>
//void StaticShade(// Output
//				 RayGMem* gOutRays,
//				 RayAuxBasic* gOutRayAux,
//				 const uint32_t maxOutRay,
//				 // Input as registers
//				 const RayReg& ray,
//				 const EmptySurface& surface,
//				 const RayAuxBasic& aux,
//				 // 
//				 RandomGPU& rng,
//				 // Input as global memory
//				 const StaticMaterialData& gMatData,
//				 const HitKey::Type& matId)
//{
//	assert(maxOutRay == 0);
//	Vector3 irrad = gOutRayAux[0].totalIrradiance;
//
//	// Material Portion
//	gOutRayAux[0].totalIrradiance = irrad * gMatData.albedo;
//
//	
//	// Ray Selection
//	RayGMem newRay = 
//	{
//
//	};
//
//
//	// TODO: Generate New Ray Directiion
//	uint32_t i = rng.Generate();
//
//	ray.
//
//}
//
//// Materials
//class StaticMaterial : public 
//	template <class TLogic, StaticMaterialData, EmptySurface,
//	StaticShade
//	class GPUMaterialGroup : public GPUMaterialGroupI
//{
//{
//
//};