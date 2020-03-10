#pragma once
/**

Tracer Partial Implementation for CUDA arcitechture

It utilizes hitting, ray memory and image memory allocation.

This is multi-gpu aware implementation.

All Tracers should inherit this class

*/

#include <deque>
#include <functional>
#include <set>
#include <map>
#include <sstream>

#include "RayLib/TracerStructs.h"
#include "RayLib/GPUTracerI.h"

#include "GPURayTracerI.h"

#include "RNGMemory.h"
#include "RayMemory.h"
#include "ImageMemory.h"

class GPUSceneI;
class CudaSystem;

struct TracerError;


class GPUTracer : public GPUTracerI
{
    private:
        // Max Bit Sizes for Efficient Sorting
        const Vector2i                      maxAccelBits;
        const Vector2i                      maxWorkBits;
        //
        const uint32_t                      maxHitSize;
        // Batches for Accelerator
        GPUBaseAcceleratorI&                baseAccelerator;
        const AcceleratorBatchMap&          accelBatches;

    protected:
        // Cuda System For Kernel Calls
        const CudaSystem&                   cudaSystem;
        // GPU Memory
        RNGMemory                           rngMemory;
        RayMemory                           rayMemory;
        ImageMemory                         imgMemory;        
        //
        TracerParameters                    params;
        //
        uint32_t                            currentRayCount;
        // Callbacks
        TracerCallbacksI*                   callbacks;
        bool                                crashed;

        // Interface
        virtual void                        ResetHitMemory(uint32_t rayCount,
                                                           HitKey baseBoundMatKey);

        void                                HitRays();
        void                                WorkRays(const WorkBatchMap&,
                                                     HitKey baseBoundMatKey);

        // Internals
        template <class... Args>
        void                                SendLog(const char*, Args...);
        void                                SendError(TracerError e, bool isFatal);

    public:
        // Constructors & Destructor
                                            GPUTracer(CudaSystem&, GPUSceneI&,
                                                      const TracerParameters&);
                                            GPUTracer(const GPUTracer&) = delete;
                                            GPUTracer(GPUTracer&&) = delete;
        GPUTracer&                          operator=(const GPUTracer&) = delete;
        GPUTracer&                          operator=(GPUTracer&&) = delete;
                                            ~GPUTracer() = default;

        // =====================//
        // RESPONSE FROM TRACER //
        // =====================//
        // Callbacks
        void                    AttachTracerCallbacks(TracerCallbacksI&) override;

        // ===================//
        // COMMANDS TO TRACER //
        // ===================//
        TracerError             Initialize() override;
        void                    SetParameters(const TracerParameters&) override;
        void                    AskParameters() override;
        void                    Finalize() override;

         // Image Related
        void                    SetImagePixelFormat(PixelFormat) override;
        void                    ReportionImage(Vector2i start = Zero2i,
                                               Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void                    ResizeImage(Vector2i resolution) override;
        void                    ResetImage() override;


};

inline void GPUTracer::AttachTracerCallbacks(TracerCallbacksI& tc)
{
    callbacks = &tc;
}

//class TracerBase : public TracerI
//{
//    private:
//        // Cuda System For Kernel Calls
//        const CudaSystem&       cudaSystem;
//
//        // Common Memory
//        RNGMemory               rngMemory;
//        RayMemory               rayMemory;
//        ImageMemory             outputImage;
//
//        // Properties
//        int                     sampleCountPerRay;
//        uint32_t                currentRayCount;
//        TracerOptions           options;
//
//        // Base tracer logic
//        TracerCallbacksI*       callbacks;
//
//        // Error related
//        bool                    healthy;
//
//        // Internals
//        template <class... Args>
//        void                    SendLog(const char*, Args...);
//        void                    SendError(TracerError e, bool isFatal);
//
//        // Fundamental Hit / Shade Loop
//        void                    HitRays();
//        void                    ShadeRays();
//
//    public:
//        // Constructors & Destructor
//                            TracerBase(CudaSystem& system);
//                            TracerBase(const TracerBase&) = delete;
//        TracerBase&         operator=(const TracerBase&) = delete;
//                            ~TracerBase() = default;
//

//
//        // ===================//
//        // COMMANDS TO TRACER //
//        // ===================//
//        // Main Calls
//        TracerError         Initialize() override;
//        void                SetOptions(const TracerOptions&) override;
//        // Requests
//        void                RequestBaseAccelerator() override;
//        void                RequestAccelerator(HitKey key) override;
//        // TODO: add sharing of other generated data (maybe interpolations etc.)
//        // and their equavilent callbacks
//
//        // Rendering Related
//        void                AttachLogic(TracerBaseLogicI&) override;
//        void                GenerateInitialRays(const GPUSceneI& scene,
//                                                int cameraId,
//                                                int samplePerLocation) override;
//        void                GenerateInitialRays(const GPUSceneI& scene,
//                                                const CPUCamera&,
//                                                int samplePerLocation) override;
//        bool                Continue() override;        // Continue hit/bounce looping (consume ray pool)
//        void                Render() override;          // Render rays  (do hit, then bounce)
//        void                FinishSamples() override;   // Finish samples (write to image)
//

//};
//
