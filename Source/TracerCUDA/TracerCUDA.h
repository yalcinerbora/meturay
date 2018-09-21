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

#include "RayLib/RNGMemory.h"
#include "RayLib/RayMemory.h"
#include "RayLib/ImageMemory.h"

#include "RayLib/TracerStructs.h"
#include "RayLib/HitStructs.h"
#include "RayLib/RayStructs.h"


//#include "RayLib/Texture.cuh"
//#include "RayLib/ArrayPortion.h"
//#include "SceneGPU.h"
//#include "ObjAcceleratorBatchI.h"
//#include "RayLib/MaterialI.h"
//
////using GPUMaterialPtr = std::unique_ptr<GPUMaterialI>;
//using ObjAcceleratorPtr = std::unique_ptr<ObjAcceleratorBatchI>;
//using MaterialRays = std::set<ArrayPortion<uint32_t>>;
//using MaterialRaysCPU = std::map<uint32_t, RayRecordCPU>;
//using MaterialHitsCPU = std::map<uint32_t, HitRecordCPU>;


struct RayAuxData
{};

class TracerCUDA : public TracerI
{
	private:
		// Callbacks
		TracerRayDelegateFunc				rayDelegateFunc;
		TracerErrorFunc						errorFunc;
		TracerAnalyticFunc					analyticFunc;
		TracerImageSendFunc					imageFunc;

		// GPU Specific Memory (Mostly Textures)
		RNGMemory							rngMemory;

		// Common Memory
		RayMemory<RayAuxData>				rayMemory;
		ImageMemory							outputImage;		
				
		// Properties
		uint32_t							currentRayCount;
		TracerParameters					parameters;
		
		// Error related
		bool								healthy;

		// Internals
		void								SendError(TracerError e, bool isFatal);
		

		// Delete These
		cudaTextureObject_t					backgroundTex;
		cudaArray_t							texArray;

	public:
		// Constructors & Destructor
										TracerCUDA();
										TracerCUDA(const TracerCUDA&) = delete;
		TracerCUDA&						operator=(const TracerCUDA&) = delete;
										~TracerCUDA() = default;

		// COMMANDS FROM TRACER
		// Delegate material ray callback
		// Tracer will use this function to send material rays to other tracers
		void					SetRayDelegateCallback(TracerRayDelegateFunc) override;
		// Error callaback for Tracer
		void					SetErrorCallback(TracerErrorFunc) override;
		// Data send callbacks
		void					SetAnalyticCallback(int sendRate, TracerAnalyticFunc) override;
		void					SetSendImageCallback(int sendRate, TracerImageSendFunc) override;
		
		// COMMANDS TO TRACER
		// Main Thread Only Calls
		void					Initialize(uint32_t seed) override;

		// Main Calls
		void					SetTime(double seconds) override;
		void					SetParams(const TracerParameters&) override;
		void					SetScene(const std::string& sceneFileName) override;
	
		// Initial Generations
		void					GenerateSceneAccelerator() override;
		void					GenerateAccelerator(uint32_t objId) override;

		// Material Related
		// Main memory bottleneck is materials.
		// Tracers are designed around this bottlenech considering GPU memory limitations.
		// A tracer will be assigned with a specific material and those rays that hits
		// to that mat will be transferred to that tracer
		void					AssignAllMaterials() override;
		void					AssignMaterial(uint32_t matId) override;
		void					LoadMaterial(uint32_t matId) override;
		void					UnloadMaterial(uint32_t matId) override;

		// Rendering
		// Generate camera rays (initialize ray pool)
		void					GenerateCameraRays(const CameraPerspective& camera,
														   const uint32_t samplePerPixel) override;

		// Continue hit/bounce looping (consume ray pool)
		bool					Continue() override;
		// Render Scene
		void					Render() override;
		// Finish samples (this mostly normalizes image with prob density etc.)
		void					FinishSamples() override;
		// Check if the tracer is crashed
		bool					IsCrashed() override;

		// Add extra rays for a specific material (from other tracers)
		// This is required because of memory limit of GPU 
		// (only specific tracers will handle specific materials)
		// Tracer will consume these rays when avaialble
		void					AddMaterialRays(const RayRecordCPU&, const HitRecordCPU&,
														uint32_t rayCount, uint32_t matId) override;

		// Image Reated
		// Set pixel format
		void					SetImagePixelFormat(PixelFormat) override;
		// Reassign a new portion of image
		void					ReportionImage(const Vector2ui& offset = Vector2ui(0, 0),
											   const Vector2ui& size = Vector2ui(0, 0)) override;
		// Resize entire image
		void					ResizeImage(const Vector2ui& resolution) override;
		// Clear image
		void					ResetImage() override;
};