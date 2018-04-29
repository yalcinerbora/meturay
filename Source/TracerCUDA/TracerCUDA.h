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
	RandomStackGMem			randomStack;
};

using GPUMaterialPtr = std::unique_ptr<GPUMaterialI>;
using ObjAcceleratorPtr = std::unique_ptr<ObjAcceleratorBatchI>;

class ImageMemory
{
	private:
		DeviceMemory	imageMem;
	protected:
	public:
		Vector3f*		imagePtr;

};

class RayMemory
{
	//private:
	//	DeviceMemory				memRayOut;
	//	DeviceMemory				memRayIn;
	//	DeviceMemory				memHit;

	//	RayRecordGMem				rayStackIn;
	//	RayRecordGMem				rayStackOut;
	//	HitRecordGMem				hitRecord;

	//	static RayRecordGMem		GenerateRayPtrs(void* mem, size_t rayCount);

	//public:
	//	// Constructors & Destructor
	//								RayMemory(size_t rayCount);
	//								RayMemory(const RayMemory&) = delete;
	//								RayMemory(RayMemory&&) = default;
	//	RayMemory&					operator=(const RayMemory&) = delete;
	//	RayMemory&					operator=(RayMemory&&) = default;
	//								~RayMemory() = default;

	//	RayRecordGMem				RayStackIn();
	//	RayRecordGMem				RayStackOut();
	//	HitRecordGMem				HitRecord();

	//	const ConstRayRecordGMem	RayStackIn() const;
	//	const ConstRayRecordGMem	RayStackOut() const;
	//	const ConstHitRecordGMem	HitRecord() const;

	//	void						Realloc();
	//	void						SwapRays();

	//	size_t						AllocatedBytes() const;
};

class TracerCUDA : public TracerI
{
	private:
		// GPU Specific Memory (Mostly Textures)
		std::vector<GPUMemory>				gpuSpecificMemory;
		std::deque<GPUMaterialPtr>			materials;

		// Common Memory
		RayMemory							rayMemory;
		// Accelerators
		std::vector<ObjAcceleratorPtr>		objAccelerators;
		// Image
		ImageMemory							image;
		// Properties
		Vector2ui							imageSegmentSize;
		Vector2ui							imageOffset;
		Vector2ui							imageResolution;

		//// Allocate
		//void								AllocateRayStack(size_t count);
		//void								AllocateRandomStack();
		//void								AllocateImage(Vector2ui resolution);
		
		// DELETE THESE MEMEBERS
//		TextureCube<float4>					backgroundTexture;
		Texture3<float4>					velocityDensityTexture;
		RandomStackGMem						random;
		uint32_t							totalRayCount;
				
	protected:
		void					SetScene(const SceneI&) override;
		void					SetParams(const TracerParameters&) override;

		void					GenerateSceneAccelerator() override;
		void					GenerateAccelerator(uint32_t objId) override;

		void					AssignAllMaterials() override;
		void					AssignMaterial(uint32_t matId) override;
		void					LoadMaterial(uint32_t matId) override;
		void					UnloadMaterial(uint32_t matId) override;

		void					GenerateCameraRays(const CameraPerspective& camera,
												   const uint32_t samplePerPixel) override;
		void					HitRays() override;
		void					GetMaterialRays(RayRecordCPU&, HitRecordCPU&,
												uint32_t rayCount, uint32_t matId) override;
		void					AddMaterialRays(const RayRecordCPU&, const HitRecordCPU&,
												uint32_t rayCount, uint32_t matId) override;
		void					BounceRays() override;
		uint32_t				RayCount() override;

		// Image Reated
		void					ReportionImage(const Vector2ui& offset = Vector2ui(0, 0),
											   const Vector2ui& size = Vector2ui(0, 0)) override;
		void					ResizeImage(const Vector2ui& resolution) override;
		void					ResetImage() override;
		std::vector<Vector3f>	GetImage() override;

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