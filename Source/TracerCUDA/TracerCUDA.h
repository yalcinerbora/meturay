#pragma once
/**

Tracer Implementation for CUDA arcitechture

(Actually all projects are pretty corralated with CUDA
this abstraction is not necessary however it is always good to
split implementation from actual call.)

This is a Threaded Implementation.
This is multi-gpu aware implementation.

Single thread will 

*/


#include <deque>
#include <functional>
#include "RayLib/TracerI.h"
#include "RayLib/DeviceMemory.h"
#include "RayLib/Texture.cuh"
#include "MaterialI.h"
#include "ObjAcceleratorBatchI.h"

struct GPUMemory
{
	TextureCube<float4> backgroundImage;
};

using GPUMaterialPtr = std::unique_ptr<GPUMaterialI>;
using ObjAcceleratorPtr = std::unique_ptr<ObjAcceleratorBatchI>;

class TracerCUDA : public TracerTI
{
	private:
		// Common Memory between GPUs (for multi-gpu systems)	
		std::deque<DeviceMemory>					deviceMemory;
		// GPU Specific Memory (Mostly Textures)
		std::vector<GPUMemory>						gpuMemory;
		std::deque<GPUMaterialPtr>					materials;
		// Common Memory
		// Ray Memory
		RayStackGMem								rayStackIn;
		RayStackGMem								rayStackOut;
		HitRecordGMem								hitRecord;
		// Accelerators
		ObjAcceleratorPtr							objAccelerators;
		
		
	protected:
		void			THRDAssignScene(const SceneI&) override;
		void			THRDSetParams(const TracerParameters&) override;

		void			THRDGenerateSceneAccelerator() override;
		void			THRDGenerateAccelerator(uint32_t objId) override;
		void			THRDAssignImageSegment(const Vector2ui& pixelStart,
											   const Vector2ui& pixelEnd) override;

		void			THRDAssignAllMaterials() override;
		void			THRDAssignMaterial(uint32_t matId) override;
		void			THRDLoadMaterial(uint32_t matId) override;
		void			THRDUnloadMaterial(uint32_t matId) override;

		void			THRDGenerateCameraRays(const CameraPerspective& camera,
											   const uint32_t samplePerPixel) override;
		void			THRDHitRays() override;
		void			THRDGetMaterialRays(const RayRecodCPU&, uint32_t rayCount, uint32_t matId) override;
		void			THRDAddMaterialRays(const ConstRayRecodCPU&, uint32_t rayCount, uint32_t matId) override;
		void			THRDBounceRays() override;
		uint32_t		THRDRayCount() override;

	public:
		// Constructors & Destructor
						TracerCUDA();
						TracerCUDA(const TracerCUDA&) = delete;
		TracerCUDA&		operator=(const TracerCUDA&) = delete;
						~TracerCUDA();

};