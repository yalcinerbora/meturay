#pragma once

#include "TracerLib/GPUPrimitiveSphere.h"
#include "TracerLib/GPUPrimitiveTriangle.h"
#include "TracerLib/GPUMaterialP.cuh"

#include "RayLib/CosineDistribution.h"

#include "TracerLogics.cuh"
#include "MaterialStructs.h"

struct BasicSurface
{
	Vector3 normal;
};

// Surface Functions
__device__ __host__
inline BasicSurface BasicSurfaceFromTri(const GPUPrimitiveTriangle::PrimitiveData& pData,
										const GPUPrimitiveTriangle::HitData& hData,
										PrimitiveId id)
{
	Vector3 n0 = pData.normalsV[id + 0];
	Vector3 n1 = pData.normalsV[id + 1];
	Vector3 n2 = pData.normalsV[id + 2];

	Vector3 baryCoord = Vector3(baryCoord[0], 
								baryCoord[1],
								1 - baryCoord[0] - baryCoord[1]);

	Vector3 nAvg = (baryCoord[0] * n0 +
					baryCoord[1] * n1 +
					baryCoord[2] * n2);
	return {nAvg};
}

__device__ __host__
inline BasicSurface BasicSurfaceFromSphr(const GPUPrimitiveSphere::PrimitiveData& pData,
										 const GPUPrimitiveSphere::HitData& hData,
										 PrimitiveId id)
{
	Vector4f data = pData.centerRadius[id + 0];
	Vector3f center = data;
	float r = data[3];

	// Convert spherical hit to cartesian
	Vector3 position = Vector3f(r * sin(hData[0]) * cos(hData[1]),
								r * sin(hData[0]) * sin(hData[1]),
								r * cos(hData[0]));

	return {(position - center).Normalize()};
}

// Shade Functions
__device__
inline void ColorMatShade(// Output
						  RayGMem* gOutRays,
						  RayAuxBasic* gOutRayAux,
						  const uint32_t maxOutRay,
						  // Input as registers
						  const RayReg& ray,
						  const BasicSurface& surface,
						  const RayAuxBasic& aux,
						  // 
						  RandomGPU& rng,
						  // Input as global memory
						  const ColorMaterialData& gMatData,
						  const HitKey::Type& matId)
{
	assert(maxOutRay == 0);
	// Inputs
	RayAuxBasic auxIn = aux;
	RayReg rayIn = ray;	
	// Outputs
	RayReg rayOut = {};
	RayAuxBasic auxOut = {};

	// Illumination Calculation
	Vector3 irrad = auxIn.totalIrradiance;
	auxOut.totalIrradiance = irrad * gMatData.dAlbedo[matId];
	
	// Material calculation is done 
	// continue to the determination of
	// ray direction over path

	// Ray Selection
	Vector3 position = rayIn.ray.AdvancedPos(rayIn.tMin);
	Vector3 normal = surface.normal;
	// Generate New Ray Directiion
	Vector2 xi(GPURand::ZeroOne<float>(rng),
			   GPURand::ZeroOne<float>(rng));
	Vector3 direction = CosineDist::HemiICDF(xi);

	// Direction vector is on surface space (hemisperical)
	// Convert it to normal oriented hemisphere
	QuatF q = QuatF::RotationBetweenZAxis(normal);
	direction = q.ApplyRotation(direction);
	
	// Write Ray
	rayOut.ray = {direction, position};
	rayOut.tMin = MathConstants::Epsilon;
	rayOut.tMax = rayIn.tMax;
	
	// All done!
	// Write to global memory
	rayOut.Update(gOutRays, 0);
	gOutRayAux[0] = auxOut;	
}

// Material Groups
class ColorMaterial final
	: public GPUMaterialGroup<TracerBasic,
							  ColorMaterialData,
							  BasicSurface,
							  ColorMatShade>
{
	public:
		static constexpr const char*	TypeName = "ConstantAlbedo";
	private:
		DeviceMemory					memory;
		ColorMaterialData				matData;

	protected:
	public:
										ColorMaterial();
										~ColorMaterial() = default;
		
		// Interface
		// Type (as string) of the primitive group
		const char*				Type() const override;
		// Allocates and Generates Data
		SceneError				InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time) override;
		SceneError				ChangeTime(const std::set<SceneFileNode>& materialNodes, double time) override;

		// Load/Unload Material			
		void					LoadMaterial(uint32_t materialId, int gpuId) override;
		void					UnloadMaterial(uint32_t material) override;
		// Material Queries
		int						InnerId(uint32_t materialId) const override;
		bool					IsLoaded(uint32_t materialId) const override;

		size_t					UsedGPUMemory() const override;
		size_t					UsedCPUMemory() const override;

		size_t					UsedGPUMemory(uint32_t materialId) const override;
		size_t					UsedCPUMemory(uint32_t materialId) const override;

		uint8_t					OutRayCount() const override;
};

// Material Batches
extern template class GPUMaterialBatch<TracerBasic,
									   ColorMaterial,
									   GPUPrimitiveTriangle,
									   BasicSurfaceFromTri>;

using ColorMatTriBatch = GPUMaterialBatch<TracerBasic,
									      ColorMaterial,
									      GPUPrimitiveTriangle,
									      BasicSurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
									   ColorMaterial,
									   GPUPrimitiveSphere,
									   BasicSurfaceFromSphr>;

using ColorMatSphrBatch = GPUMaterialBatch<TracerBasic,
									       ColorMaterial,
									       GPUPrimitiveSphere,
									       BasicSurfaceFromSphr>;