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
#include "TracerError.h"
#include "Types.h"

class SceneI;
struct CameraPerspective;
struct TracerParameters;
struct HitRecordCPU;
struct RayRecordCPU;
struct TracerAnalyticData;

// Callbacks for tracer
typedef void(*TracerRayDelegateFunc)(const RayRecordCPU&, const HitRecordCPU&,
									 uint32_t rayCount, uint32_t matId);

typedef void(*TracerErrorFunc)(TracerError);
typedef void(*TracerAnalyticFunc)(TracerAnalyticData);
typedef void(*TracerImageSendFunc)(const Vector2ui& offset, 
								   const Vector2ui& size, 
								   const std::vector<float> imagePortion);

class TracerI
{
	public:
		virtual							~TracerI() = default;

		// COMMANDS FROM TRACER
		// Delegate material ray callback
		// Tracer will use this function to send material rays to other tracers
		virtual void					SetRayDelegateCallback(TracerRayDelegateFunc) = 0;
		// Error callaback for Tracer
		virtual void					SetErrorCallback(TracerErrorFunc) = 0;
		// Data send callbacks
		virtual void					SetAnalyticCallback(int sendRate, TracerAnalyticFunc) = 0;
		virtual void					SetSendImageCallback(int sendRate, TracerImageSendFunc) = 0;
		
		// COMMANDS TO TRACER
		virtual void					Initialize(uint32_t seed) = 0;

		// Main Calls
		virtual void					SetTime(double seconds) = 0;
		virtual void					SetParams(const TracerParameters&) = 0;
		virtual void					SetScene(const std::string& sceneFileName) = 0;
	
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
		// Generate camera rays (initialize ray pool)
		virtual void					GenerateCameraRays(const CameraPerspective& camera,
														   const uint32_t samplePerPixel) = 0;

		// Continue hit/bounce looping (consume ray pool)
		virtual bool					Continue() = 0;
		// Render rays
		virtual void					Render() = 0;
		// Finish samples (this mostly normalizes image with prob density etc.)
		virtual void					FinishSamples() = 0;
		// Check if the tracer is crashed
		virtual bool					IsCrashed() = 0;

		// Add extra rays for a specific material (from other tracers)
		// This is required because of memory limit of GPU 
		// (only specific tracers will handle specific materials)
		// Tracer will consume these rays when avaialble
		virtual void					AddMaterialRays(const RayRecordCPU&, const HitRecordCPU&,
														uint32_t rayCount, uint32_t matId) = 0;

		// Image Reated
		// Set pixel format		
		virtual void					SetImagePixelFormat(PixelFormat) = 0;
		// Reassign a new portion of image
		virtual void					ReportionImage(const Vector2ui& offset = Vector2ui(0, 0),
													   const Vector2ui& size = Vector2ui(0, 0)) = 0;
		// Resize entire image
		virtual void					ResizeImage(const Vector2ui& resolution) = 0;
		// Clear image
		virtual void					ResetImage() = 0;
};