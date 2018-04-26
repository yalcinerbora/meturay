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
#include "RayLib/RandomStructs.h"

#include "ObjAcceleratorBatchI.h"
#include "MaterialI.h"

struct GPUMemory
{
	//TextureCube<float4> backgroundImage;
	RandomStackGMem		randomStack;
};

using GPUMaterialPtr = std::unique_ptr<GPUMaterialI>;
using ObjAcceleratorPtr = std::unique_ptr<ObjAcceleratorBatchI>;

class TracerCUDA : public TracerTI
{
	private:
		// Common Memory between GPUs (for multi-gpu systems)	
		std::deque<DeviceMemory>					commonMemory;
		// GPU Specific Memory (Mostly Textures)
		std::vector<GPUMemory>						gpuSpecificMemory;
		std::deque<GPUMaterialPtr>					materials;
		
		// Common Memory
		// Ray Memory
		RayStackGMem								rayStackIn;
		RayStackGMem								rayStackOut;
		HitRecordGMem								hitRecord;
		// Accelerators
		ObjAcceleratorPtr							objAccelerators;
		
		// Allocate
		void										AllocateRayStack(size_t count);
		void										AllocateRandomStack();
		void										AllocateImage(Vector2ui resolution);

		Vector3f*									dOutImage;
		
		// DELETE THESE MEMEBERS
//		TextureCube<float4>							backgroundTexture;
		Texture3<float4>							velocityDensityTexture;
		RandomStackGMem								random;
		uint32_t									totalRayCount;
				
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

		// Main Thread Only Calls
		void			Initialize() override;





		// DELETE THOSE
		void					LoadBackgroundCubeMap(const std::vector<float>& cubemap) override;
		void					LoadFluidToGPU(const std::vector<float>& velocityDensity,
											   const Vector3ui& size) override;
		void					CS568GenerateCameraRays(const CameraPerspective& cam,
																const Vector2ui resolution,
																const uint32_t samplePerPixel) override;
		void					LaunchRays(const Vector3f& backgroundColor, 
										   const Vector3ui& textureSize,
										   
										   const Vector3f& bottomLeft,
										   const Vector3f& length) override;
		std::vector<Vector3f>	GetImage(const Vector2ui& resolution) override;

};