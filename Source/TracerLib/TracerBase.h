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
#include "RayLib/TracerI.h"

#include "RNGMemory.h"
#include "RayMemory.h"
#include "ImageMemory.h"

class TracerBaseLogicI;
class TracerLogicGeneratorI;
struct TracerError;

class TracerBase : public TracerI
{
	private:
		// Common Memory
		RNGMemory			rngMemory;
		RayMemory			rayMemory;
		ImageMemory			outputImage;		

		// Properties
		uint32_t				currentRayCount;
		TracerOptions		options;

		// Base tracer logic
		TracerBaseLogicI*	currentLogic;
		TracerCallbacksI*	callbacks;

		// Error related
		bool					healthy;

		// Internals
		void					SendError(TracerError e, bool isFatal);
		void					HitRays();
		void					ShadeRays();

	public:
		// Constructors & Destructor
							TracerBase();
							TracerBase(const TracerBase&) = delete;
		TracerBase&			operator=(const TracerBase&) = delete;
							~TracerBase() = default;

		// =====================//
		// RESPONSE FROM TRACER //
		// =====================//
		// Callbacks
		void				AttachTracerCallbacks(TracerCallbacksI&) override;

		// ===================//
		// COMMANDS TO TRACER //
		// ===================//	
		// Main Calls
		void				Initialize(int leaderGPUId = 0) override;
		void				SetOptions(const TracerOptions&) override;
		// Requests
		void				RequestBaseAccelerator() override;
		void				RequestAccelerator(HitKey key) override;
		// TODO: add sharing of other generated data (maybe interpolations etc.)
		// and their equavilent callbacks

		// Rendering Related
		void				AttachLogic(TracerBaseLogicI&) override;
		void				GenerateInitialRays(const GPUScene& scene,
											int cameraId,
											int samplePerLocation) override;
		bool				Continue() override;			// Continue hit/bounce looping (consume ray pool)
		void				Render() override;			// Render rays	(do hit, then bounce)		
		void				FinishSamples() override;	// Finish samples (write to image)

		// Image Reated
		void				SetImagePixelFormat(PixelFormat) override;
		void				ReportionImage(Vector2i start = Zero2i,
									   Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
		void				ResizeImage(Vector2i resolution) override;
		void				ResetImage() override;
};

inline void TracerBase::AttachTracerCallbacks(TracerCallbacksI& tc)
{
	callbacks = &tc;
}