#pragma once
/**


*/

#include <string>
#include <cstdint>
#include <set>

#include "HitStructs.cuh"

struct RayGMem;
struct SceneFileNode;
struct SceneError;

class RNGMemory;
class GPUPrimitiveGroupI;

// Defines the same type materials
// Logics consists of loading unloading certain material
// This struct holds the material data in a batched fashion (textures arrays etc)
// These are singular and can be shared by multiple accelrator batches
class GPUMaterialGroupI
{
	public:
		virtual								~GPUMaterialGroupI() = default;

		// Interface
		// Type (as string) of the primitive group
		virtual const char*					Type() const = 0;
		// Allocates and Generates Data
		virtual SceneError					InitializeGroup(const std::set<SceneFileNode>& materialNodes, double time) = 0;
		virtual SceneError					ChangeTime(const std::set<SceneFileNode>& materialNodes, double time) = 0;

		// Load/Unload Material				
		virtual void						LoadMaterial(uint32_t materialId, int gpuId) = 0;
		virtual void						UnloadMaterial(uint32_t material) = 0;
		// Material Queries
		virtual int							InnerId(uint32_t materialId) const = 0;
		virtual bool						IsLoaded(uint32_t materialId) const = 0;

		virtual size_t						UsedGPUMemory() const = 0;
		virtual size_t						UsedCPUMemory() const = 0;
		virtual size_t						UsedGPUMemory(uint32_t materialId) const = 0;
		virtual size_t						UsedCPUMemory(uint32_t materialId) const = 0;

		virtual uint8_t						OutRayCount() const = 0;
};

// Defines call group over a certain material group
// The batch further specializes over a primitive logic
// which defines how certain primitive data could be fetched
class GPUMaterialBatchI
{
	public:
		virtual								~GPUMaterialBatchI() = default;

		// Interface
		// Type (as string) of the primitive group
		virtual const char*					Type() const = 0;
		// Kernel Call
		virtual void						ShadeRays(// Output
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
													  RNGMemory& rngMem) const = 0;

		// Every MaterialBatch is available for a specific primitive / material data
		virtual const GPUPrimitiveGroupI&				PrimitiveGroup() const = 0;
		virtual const GPUMaterialGroupI&				MaterialGroup() const = 0;

		virtual uint8_t									OutRayCount() const = 0;
};

class GPUBoundaryMatGroupI : public GPUMaterialGroupI
{
	public:
		virtual							~GPUBoundaryMatGroupI() = default;

		// Interface
		virtual void					AttachOutputImage(Vector4* pixels) = 0;
};