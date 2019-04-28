#pragma once

#include "Surfaces.h"

#include "RayLib/CosineDistribution.h"

#include "TracerLib/GPUMaterialP.cuh"

__device__
inline void GIAlbedoMatShade(// Output
							 Vector4f* gImage,
							 //
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
							 const ConstantAlbedoMatData& gMatData,
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
	Vector3 rad = auxIn.totalRadiance;
	auxOut.totalRadiance = rad * gMatData.dAlbedo[matId];
	// Material calculation is done 
	// continue to the determination of
	// ray direction over path

	// Ray Selection
	Vector3 position = rayIn.ray.AdvancedPos(rayIn.tMax);
	Vector3 normal = surface.normal;
	// Generate New Ray Directiion
	Vector2 xi(GPURand::ZeroOne<float>(rng),
			   GPURand::ZeroOne<float>(rng));
	Vector3 direction = CosineDist::HemiICDF(xi);

	// Direction vector is on surface space (hemisperical)
	// Convert it to normal oriented hemisphere
	QuatF q = QuatF::RotationBetweenZAxis(normal);
	direction = q.ApplyRotation(direction);

	// Advance slightly to prevent self intersection
	position += direction * MathConstants::Epsilon;

	// Write Ray
	rayOut.ray = {direction, position};
	rayOut.tMin = 0.0f;
	rayOut.tMax = INFINITY;

	// All done!
	// Write to global memory
	rayOut.Update(gOutRays, 0);
	gOutRayAux[0] = auxOut;
}

class GIAlbedoMat final
	: public GPUMaterialGroup<TracerBasic,
							  ConstantAlbedoMatData,
							  BasicSurface,
							  GIAlbedoMatShade>
{
	public:
		static constexpr const char*	TypeName = "GIAlbedo";

	private:
		DeviceMemory				memory;
		ConstantAlbedoMatData	matData;

	protected:
	public:
								GIAlbedoMat(int gpuId);
								~GIAlbedoMat() = default;
		
		// Interface
		// Type (as string) of the primitive group
		const char*				Type() const override;
		// Allocates and Generates Data
		SceneError				InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time) override;
		SceneError				ChangeTime(const std::set<SceneFileNode>& materialNodes, double time) override;

		// Material Queries
		int						InnerId(uint32_t materialId) const override;
		bool						IsLoaded(uint32_t materialId) const override;

		size_t					UsedGPUMemory() const override;
		size_t					UsedCPUMemory() const override;

		size_t					UsedGPUMemory(uint32_t materialId) const override;
		size_t					UsedCPUMemory(uint32_t materialId) const override;

		uint8_t					OutRayCount() const override;
};

// Mat Batch Extern
extern template class GPUMaterialBatch<TracerBasic,
									   GIAlbedoMat,
									   GPUPrimitiveTriangle,
									   BasicSurfaceFromTri>;

using GIAlbedoTriBatch = GPUMaterialBatch<TracerBasic,
										  GIAlbedoMat,
										  GPUPrimitiveTriangle,
										  BasicSurfaceFromTri>;

extern template class GPUMaterialBatch<TracerBasic,
									   GIAlbedoMat,
									   GPUPrimitiveSphere,
									   BasicSurfaceFromSphr>;

using GIAlbedoSphrBatch = GPUMaterialBatch<TracerBasic,
									       GIAlbedoMat,
									       GPUPrimitiveSphere,
									       BasicSurfaceFromSphr>;