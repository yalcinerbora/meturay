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

#include "RayLib/TracerStructs.h"
#include "TracerI.h"

#include "RNGMemory.h"
#include "RayMemory.h"
#include "ImageMemory.h"

class TracerBaseLogicI;
class TracerLogicGeneratorI;
struct TracerError;

class TracerBase : public TracerI
{
	private:
		// Callbacks
		TracerRayDelegateFunc			rayDelegateFunc;
		TracerErrorFunc					errorFunc;
		TracerAnalyticFunc				analyticFunc;
		TracerImageSendFunc				imageFunc;
		TracerAcceleratorSendFunc		accSendFunc;
		TracerBaseAcceleratorSendFunc	baseSendFunc;

		// Common Memory
		RNGMemory						rngMemory;
		RayMemory						rayMemory;
		ImageMemory						outputImage;		
			
		// Properties
		uint32_t						currentRayCount;
		TracerParameters				parameters;
	
		// Base tracer logic
		TracerBaseLogicI*				tracerSystem;
		TracerLogicGeneratorI*			logicGenerator;

		// Error related
		bool							healthy;

		// Internals
		void							SendError(TracerError e, bool isFatal);
		void							HitRays();
		void							SendAndRecieveRays();
		void							ShadeRays();

	public:
		// Constructors & Destructor
										TracerBase();
										TracerBase(const TracerBase&) = delete;
		TracerBase&						operator=(const TracerBase&) = delete;
										~TracerBase() = default;

		// =====================//
		// RESPONSE FROM TRACER //
		// =====================//
		// Delegate material ray callback
		// Tracer will use this function to send material rays to other tracers
		void					SetRayDelegateCallback(TracerRayDelegateFunc) override;
		// Error callaback for Tracer
		void					SetErrorCallback(TracerErrorFunc) override;
		// Data send callbacks
		void					SetAnalyticCallback(int sendRate, TracerAnalyticFunc) override;
		void					SetSendImageCallback(int sendRate, TracerImageSendFunc) override;
		// Accelerator sharing
		void					SetSendAcceleratorCallback(TracerAcceleratorSendFunc) override;
		void					SetSendBaseAcceleratorCallback(TracerBaseAcceleratorSendFunc) override;

		// ===================//
		// COMMANDS TO TRACER //
		// ===================//
		// Main Thread Only Calls
		void					Initialize(uint32_t seed, TracerBaseLogicI&) override;

		// Main Calls
		void					SetTime(double seconds) override;
		void					SetParams(const TracerParameters&) override;
		void					SetScene(const std::string& sceneFileName) override;
	
		// Requests
		void					RequestBaseAccelerator() override;
		void					RequestAccelerator(int key) override;
		// TODO: add sharing of other generated data (maybe interpolations etc.)
		// and their equavilent callbacks

		// Material Related
		// Main memory bottleneck is materials.
		// Tracers are designed around this bottleneck considering GPU memory limitations.
		// A tracer will be assigned with a specific material and those rays that hits
		// to that mat will be transferred to that tracer
		void					AssignAllMaterials() override;
		void					AssignMaterial(uint32_t matId) override;
		void					UnassignAllMaterials() override;
		void					UnassignMaterial(uint32_t matId) override;

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
		void					AddMaterialRays(const RayCPU&, const HitCPU&,
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

inline void TracerBase::SetRayDelegateCallback(TracerRayDelegateFunc f)
{
	rayDelegateFunc = f;
}

inline void TracerBase::SetErrorCallback(TracerErrorFunc f)
{
	errorFunc = f;
}

inline void TracerBase::SetAnalyticCallback(int sendRate, TracerAnalyticFunc f)
{
	analyticFunc = f;
}

inline void TracerBase::SetSendImageCallback(int sendRate, TracerImageSendFunc f)
{
	imageFunc = f;
}

inline void TracerBase::SetSendAcceleratorCallback(TracerAcceleratorSendFunc f)
{
	accSendFunc = f;
}

inline void TracerBase::SetSendBaseAcceleratorCallback(TracerBaseAcceleratorSendFunc f)
{
	baseSendFunc = f;
}