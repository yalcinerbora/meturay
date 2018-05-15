#pragma once

/**

Tracer Interface

Main Interface for Tracer DLLs. Only GPU tracer will be
implemented, still tracer is interfaced for further implementations

Tracer Interface is a threaded interface (which means that it repsesents a thread)
which does send commands to GPU to do ray tracing 
(it is responsible for utilizing all GPUs on the computer). 



*/

#include <cstdint>
#include <future>
#include "Vector.h"
#include "RayHitStructs.h"

class SceneI;
struct CameraPerspective;
struct TracerParameters;

class TracerI
{
	public:
		virtual							~TracerI() = default;

		// Main Thread Only Calls
		virtual void					Initialize() = 0;

		// Main Calls
		virtual void					SetTime(double seconds) = 0;
		virtual void					SetScene(const std::string& sceneFileName) = 0;
		virtual void					SetParams(const TracerParameters&) = 0;
	
		// Initial Generations
		virtual void					GenerateSceneAccelerator() = 0;
		virtual void					GenerateAccelerator(uint32_t objId) = 0;

		// Material Related
		// Main memory bottleneck is materials.
		// Tracers are designed around this bottlenech considering GPU memory limitations.
		// A tracer will be assigned with a specific material and those rays that hits
		// to that mat will be transferred to that tracer
		virtual void					AssignAllMaterials() = 0;
		virtual void					AssignMaterial(uint32_t matId) = 0;
		virtual void					LoadMaterial(uint32_t matId) = 0;
		virtual void					UnloadMaterial(uint32_t matId) = 0;

		// Rendering
		// Loop HitRays/BounceRays until ray count is zero
		// Transfer Material rays between tracer nodes using Get/AddMaterialRays
		virtual void					GenerateCameraRays(const CameraPerspective& camera,
														   const uint32_t samplePerPixel) = 0;
		virtual void					HitRays() = 0;		
		virtual void					GetMaterialRays(RayRecordCPU&, HitRecordCPU&, 
														uint32_t rayCount, uint32_t matId) = 0;
		virtual void					AddMaterialRays(const RayRecordCPU&, const HitRecordCPU&,
														uint32_t rayCount, uint32_t matId) = 0;
		virtual void					BounceRays() = 0;
		virtual uint32_t				RayCount() = 0;

		// Image Reated
		virtual void					ReportionImage(const Vector2ui& offset = Vector2ui(0, 0),
													   const Vector2ui& size = Vector2ui(0, 0)) = 0;
		virtual void					ResizeImage(const Vector2ui& resolution) = 0;
		virtual void					ResetImage() = 0;
		virtual std::vector<Vector3f>	GetImage() = 0;
		
		//// DELETE THOSE THOSE ARE FOR FAST BAREBONES EXECUTION
		//virtual void					LoadBackgroundCubeMap(const std::vector<float>& cubemap) = 0;
		//virtual void					LoadFluidToGPU(const std::vector<float>& velocityDensity,
		//											   const Vector3ui& size) = 0;
		//virtual void					CS568GenerateCameraRays(const CameraPerspective& cam,
		//														const Vector2ui resolution,
		//														const uint32_t samplePerPixel) = 0;
		//virtual void					LaunchRays(const Vector3f& backgroundColor,
		//										   const Vector3ui& textureSize,
		//										   const Vector3f& bottomLeft,
		//										   const Vector3f& length) = 0;
		//virtual std::vector<Vector3f>	GetImage(const Vector2ui& resolution) = 0;
};
