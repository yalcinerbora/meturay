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
#include <set>
#include <map>

#include "RayLib/TracerI.h"
#include "RayLib/DeviceMemory.h"
#include "RayLib/Texture.cuh"
#include "RayLib/RandomStructs.h"
#include "RayLib/TracerStructs.h"
#include "RayLib/ArrayPortion.h"

#include "SceneGPU.h"

#include "ObjAcceleratorBatchI.h"
#include "RayLib/MaterialI.h"

//using GPUMaterialPtr = std::unique_ptr<GPUMaterialI>;
using ObjAcceleratorPtr = std::unique_ptr<ObjAcceleratorBatchI>;
using MaterialRays = std::set<ArrayPortion<uint32_t>>;
using MaterialRaysCPU = std::map<uint32_t, RayRecordCPU>;
using MaterialHitsCPU = std::map<uint32_t, HitRecordCPU>;

class ImageMemory
{
	private:
		DeviceMemory			imageMem;

		Vector2ui				imageSegmentSize;
		Vector2ui				imageOffset;
		Vector2ui				imageResolution;

		Vector3f*				imagePtr;

	protected:
	public:
		// Constructors & Destructors
								ImageMemory();
								ImageMemory(const ImageMemory&) = delete;
		ImageMemory&			operator=(const ImageMemory&) = delete;
								~ImageMemory() = default;
					
		void					ReportionImage(const Vector2ui& offset,
											   const Vector2ui& size);
		void					ResizeImage(const Vector2ui& resolution);
		void					ResetImage();
		std::vector<Vector3f>	GetImage();

		// Getters
		Vector2ui				ImageSegment() const;
		Vector2ui				ImageOffset() const;
		Vector2ui				ImageResolution() const;
		Vector3f*				ImageGMem();
};

class RNGMemory
{
	private:
		DeviceMemory						memRandom;
		std::vector<RandomStackGMem>		randomStacks;

	protected:
	public:
		// Constructors & Destructor
											RNGMemory() = default;
											RNGMemory(uint32_t seed);
											RNGMemory(const RNGMemory&) = delete;
											RNGMemory(RNGMemory&&) = default;
		RNGMemory&							operator=(const RNGMemory&) = delete;
		RNGMemory&							operator=(RNGMemory&&) = default;
											~RNGMemory() = default;

		RandomStackGMem						RandomStack(uint32_t gpuId);
		size_t								SharedMemorySize(uint32_t gpuId);
};

class RayMemory
{
	private:
		DeviceMemory				memRayIn;
		DeviceMemory				memRayOut;		
		DeviceMemory				memHit;

		RayRecordGMem				rayStackIn;
		RayRecordGMem				rayStackOut;
		HitRecordGMem				hitRecord;

		static RayRecordGMem		GenerateRayPtrs(void* mem, size_t rayCount);
		static HitRecordGMem		GenerateHitPtrs(void* mem, size_t rayCount);
		static size_t				TotalMemoryForRay(size_t rayCount);
		static size_t				TotalMemoryForHit(size_t rayCount);

	public:
		// Constructors & Destructor
									RayMemory();
									RayMemory(const RayMemory&) = delete;
									RayMemory(RayMemory&&) = default;
		RayMemory&					operator=(const RayMemory&) = delete;
		RayMemory&					operator=(RayMemory&&) = default;
									~RayMemory() = default;

		// Accessors
		RayRecordGMem				RayStackIn();
		RayRecordGMem				RayStackOut();
		HitRecordGMem				HitRecord();

		const ConstRayRecordGMem	RayStackIn() const;
		const ConstRayRecordGMem	RayStackOut() const;
		const ConstHitRecordGMem	HitRecord() const;
		
		// Memory Arrangement
		void						AllocForCameraRays(size_t rayCount);
		MaterialRays				ResizeRayIn(const MaterialRays& current,
												const MaterialRays& external,
												const MaterialRaysCPU& rays,
												const MaterialHitsCPU& hits);
		void						ResizeRayOut(size_t rayCount);
		void						SwapRays(size_t rayCount);
};

class TracerCUDA : public TracerI
{
	private:
		// GPU Specific Memory (Mostly Textures)
		RNGMemory							rngMemory;		
		//std::deque<GPUMaterialPtr>		materials;
		//std::vector<GPUMemory>			gpuSpecificMemory;

		// Common Memory
		RayMemory							rayMemory;
		ImageMemory							outputImage;		
		std::vector<ObjAcceleratorPtr>		objAccelerators;

		// Scene (Also common memory)
		SceneGPU							scene;

		// Properties
		uint32_t							sampleCount;
		uint32_t							currentRayCount;
		TracerParameters					parameters;
		
		// Portioning of Material Rays
		MaterialRays						materialRayPortions;
		
		// Error Callback
		ErrorCallbackFunction				errorFunc;

		// Internals
		void								SortRaysByMaterial();
		void								SortRaysBySurface();

		// Delete These
		cudaTextureObject_t					backgroundTex;
		cudaArray_t							texArray;

	public:
		// Constructors & Destructor
										TracerCUDA();
										TracerCUDA(const TracerCUDA&) = delete;
		TracerCUDA&						operator=(const TracerCUDA&) = delete;
										~TracerCUDA();

		// Main Thread Only Calls
		void							Initialize(uint32_t seed) override;
		virtual void					SetErrorCallback(ErrorCallbackFunction) override;

		// Main Calls
		void							SetTime(double seconds) override;
		void							SetScene(const std::string& sceneFileName) override;
		void							SetParams(const TracerParameters&) override;

		void							GenerateSceneAccelerator() override;
		void							GenerateAccelerator(uint32_t objId) override;

		void							AssignAllMaterials() override;
		void							AssignMaterial(uint32_t matId) override;
		void							LoadMaterial(uint32_t matId) override;
		void							UnloadMaterial(uint32_t matId) override;

		void							GenerateCameraRays(const CameraPerspective& camera,
														   const uint32_t samplePerPixel) override;
		void							HitRays(int frame) override;
		void							GetMaterialRays(RayRecordCPU&, HitRecordCPU&,
														uint32_t rayCount, uint32_t matId) override;
		void							AddMaterialRays(const RayRecordCPU&, const HitRecordCPU&,
														uint32_t rayCount, uint32_t matId) override;
		void							BounceRays() override;
		uint32_t						RayCount() override;

		// Image Reated
		void							ReportionImage(const Vector2ui& offset = Vector2ui(0, 0),
													   const Vector2ui& size = Vector2ui(0, 0)) override;
		void							ResizeImage(const Vector2ui& resolution) override;
		void							ResetImage() override;
		std::vector<Vector3f>			GetImage() override;

		//// DELETE THOSE
		//void							LoadBackgroundCubeMap(const std::vector<float>& cubemap) override;
		//void							LoadFluidToGPU(const std::vector<float>& velocityDensity,
		//											   const Vector3ui& size) override;
		//void							CS568GenerateCameraRays(const CameraPerspective& cam,
		//																const Vector2ui resolution,
		//																const uint32_t samplePerPixel) override;
		//void							LaunchRays(const Vector3f& backgroundColor, 
		//										   const Vector3ui& textureSize,
		//										   
		//										   const Vector3f& bottomLeft,
		//										   const Vector3f& length) override;
		//std::vector<Vector3f>			GetImage(const Vector2ui& resolution) override;

};