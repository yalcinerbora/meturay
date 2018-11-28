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
#include <vector>

#include "RayLib/Vector.h"
#include "RayLib/Types.h"

struct HitCPU;
struct RayCPU;
struct TracerError;
struct TracerOptions;

// Callback
struct CameraPerspective;
struct TracerAnalyticData;

// Main Tracer Logicc
class TracerBaseLogicI;

// Callbacks for tracer
typedef void(*TracerRayDelegateFunc)(const RayCPU&, const HitCPU&,
									 uint32_t rayCount, uint32_t matId);
typedef void(*TracerErrorFunc)(TracerError);
typedef void(*TracerAnalyticFunc)(TracerAnalyticData);
typedef void(*TracerImageSendFunc)(const Vector2ui& offset,
								   const Vector2ui& size,
								   const std::vector<float> imagePortion);
typedef void(*TracerAcceleratorSendFunc)(int key, const std::vector<Byte> data);
typedef void(*TracerBaseAcceleratorSendFunc)(const std::vector<Byte> data);

class TracerI
{
	public:
		virtual							~TracerI() = default;

		// =====================//
		// RESPONSE FROM TRACER //
		// =====================//
		// Delegate material ray callback
		// Tracer will use this function to send material rays to other tracers
		virtual void					SetRayDelegateCallback(TracerRayDelegateFunc) = 0;
		// Error callaback for Tracer
		virtual void					SetErrorCallback(TracerErrorFunc) = 0;
		// Data send callbacks
		virtual void					SetAnalyticCallback(int sendRate, TracerAnalyticFunc) = 0;
		virtual void					SetSendImageCallback(int sendRate, TracerImageSendFunc) = 0;
		// Accelerator sharing
		virtual void					SetSendAcceleratorCallback(TracerAcceleratorSendFunc) = 0;
		virtual void					SetSendBaseAcceleratorCallback(TracerBaseAcceleratorSendFunc) = 0;
		
		// ===================//
		// COMMANDS TO TRACER //
		// ===================//
		virtual void					Initialize(TracerBaseLogicI&) = 0;

		// Main Calls
		virtual void					SetOptions(const TracerOptions&) = 0;
	
		// Requests
		virtual void					RequestBaseAccelerator() = 0;
		virtual void					RequestAccelerator(int key) = 0;
		// TODO: add sharing of other generated data (maybe interpolations etc.)
		// and their equavilent callbacks

		// Material Related
		// Main memory bottleneck is materials.
		// Tracers are designed around this bottlenech considering GPU memory limitations.
		// A tracer will be assigned with a specific material and those rays that hits
		// to that mat will be transferred to that tracer
		virtual void					AssignAllMaterials() = 0;
		virtual void					AssignMaterial(uint32_t matId) = 0;
		virtual void					UnassignAllMaterials() = 0;
		virtual void					UnassignMaterial(uint32_t matId) = 0;

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
		virtual void					AddMaterialRays(const RayCPU&, const HitCPU&,
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