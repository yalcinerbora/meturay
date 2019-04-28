#pragma once

/**

Tracer Interface

Main Interface for Tracer DLLs. Only GPU tracer will be
implemented, still tracer is interfaced for further implementations

Tracer Interface is a threaded interface (which means that it repsesents a thread)
which does send commands to GPU to do ray tracing 
(it is responsible for utilizing all GPUs on the computer). 



*/

#include "Vector.h"
#include "Types.h"
#include "Constants.h"
#include "HitStructs.h"

struct TracerError;
struct TracerOptions;

class GPUScene;

// Main Tracer Logicc
class TracerCallbacksI;
class TracerBaseLogicI;

class TracerI
{
	public:
		virtual							~TracerI() = default;

		// =====================//
		// RESPONSE FROM TRACER //
		// =====================//
		// Callbacks
		virtual void					AttachTracerCallbacks(TracerCallbacksI&) = 0;
				
		// ===================//
		// COMMANDS TO TRACER //
		// ===================//		
		// Main Calls
		virtual TracerError			Initialize(int leaderGPUId = 0)  = 0;
		virtual void					SetOptions(const TracerOptions&) = 0;	
		// Requests
		virtual void					RequestBaseAccelerator() = 0;
		virtual void					RequestAccelerator(HitKey key) = 0;
		// TODO: add sharing of other generated data (maybe interpolations etc.)
		// and their equavilent callbacks

		// Rendering Related
		virtual void					AttachLogic(TracerBaseLogicI&) = 0;
		virtual void					GenerateInitialRays(const GPUScene& scene, 
															int cameraId,
															int samplePerLocation) = 0;
		virtual bool					Continue() = 0;			// Continue hit/bounce looping (consume ray pool)
		virtual void					Render() = 0;			// Render rays	(do hit, then bounce)		
		virtual void					FinishSamples() = 0;	// Finish samples (write to image)

		// Image Reated		
		virtual void					SetImagePixelFormat(PixelFormat) = 0;
		virtual void					ReportionImage(Vector2i start = Zero2i,
													   Vector2i end = BaseConstants::IMAGE_MAX_SIZE) = 0;
		virtual void					ResizeImage(Vector2i resolution) = 0;
		virtual void					ResetImage() = 0;
};