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
#include <sstream>

#include "RayLib/TracerStructs.h"

#include "GPURayTracerI.h"

#include "RNGMemory.h"
#include "RayMemory.h"
#include "ImageMemory.h"

class CudaSystem;
class TracerLogicGeneratorI;
struct TracerError;

class TracerBase
{
    private:

    protected:
        // Cuda System For Kernel Calls
        const CudaSystem&                           cudaSystem;
        // Max Bit Sizes for Efficient Sorting
        const Vector2i                              maxAccelBits;
        const Vector2i                              maxWorkBits;
        //        
        const uint32_t                              maxHitSize;
        // GPU Memory
        RNGMemory                                   rngMemory;
        RayMemory                                   rayMemory;
        // Batches for Accelerator
        GPUBaseAcceleratorI&                        baseAccelerator;
        AcceleratorBatchMappings&                   accelBatches;
        //
        const TracerParameters                      params;
        //
        uint32_t                                    currentRayCount;

        // Interface
        // Initialize and allocate for rays
        virtual TracerError                         Initialize() = 0;
        virtual void                                ResetHitMemory(uint32_t rayCount,
                                                                   HitKey baseBoundMatKey);

        virtual void                                HitRays() = 0;
        virtual void                                WorkRays(const WorkBatchMappings&,
                                                             HitKey baseBoundMatKey);

    public:
        // Constructors & Destructor
                                                    TracerBase(CudaSystem&,
                                                               // Accelerators that are required
                                                               // for hit loop
                                                               GPUBaseAcceleratorI&,
                                                               AcceleratorBatchMappings&,
                                                               // Bits for sorting
                                                               const Vector2i maxAccelBits,
                                                               const Vector2i maxWorkBits,
                                                               // Hit size for union allocation
                                                               const uint32_t maxHitSize,
                                                               // Initialization Param of tracer
                                                               const TracerParameters&);
                                                    TracerBase(const TracerBase&) = delete;
                                                    TracerBase(TracerBase&&) = delete;
        TracerBase&                                 operator=(const TracerBase&) = delete;
        TracerBase&                                 operator=(TracerBase&&) = delete;
                                                    ~TracerBase() = default;


};

class TracerBase : public TracerI
{
    private:
        // Cuda System For Kernel Calls
        const CudaSystem&       cudaSystem;

        // Common Memory
        RNGMemory               rngMemory;
        RayMemory               rayMemory;
        ImageMemory             outputImage;

        // Properties
        int                     sampleCountPerRay;
        uint32_t                currentRayCount;
        TracerOptions           options;

        // Base tracer logic
        TracerCallbacksI*       callbacks;

        // Error related
        bool                    healthy;

        // Internals
        template <class... Args>
        void                    SendLog(const char*, Args...);
        void                    SendError(TracerError e, bool isFatal);

        // Fundamental Hit / Shade Loop
        void                    HitRays();
        void                    ShadeRays();

    public:
        // Constructors & Destructor
                            TracerBase(CudaSystem& system);
                            TracerBase(const TracerBase&) = delete;
        TracerBase&         operator=(const TracerBase&) = delete;
                            ~TracerBase() = default;

        // =====================//
        // RESPONSE FROM TRACER //
        // =====================//
        // Callbacks
        void                AttachTracerCallbacks(TracerCallbacksI&) override;

        // ===================//
        // COMMANDS TO TRACER //
        // ===================//
        // Main Calls
        TracerError         Initialize() override;
        void                SetOptions(const TracerOptions&) override;
        // Requests
        void                RequestBaseAccelerator() override;
        void                RequestAccelerator(HitKey key) override;
        // TODO: add sharing of other generated data (maybe interpolations etc.)
        // and their equavilent callbacks

        // Rendering Related
        void                AttachLogic(TracerBaseLogicI&) override;
        void                GenerateInitialRays(const GPUSceneI& scene,
                                                int cameraId,
                                                int samplePerLocation) override;
        void                GenerateInitialRays(const GPUSceneI& scene,
                                                const CameraPerspective&,
                                                int samplePerLocation) override;
        bool                Continue() override;        // Continue hit/bounce looping (consume ray pool)
        void                Render() override;          // Render rays  (do hit, then bounce)
        void                FinishSamples() override;   // Finish samples (write to image)

        // Image Related
        void                SetImagePixelFormat(PixelFormat) override;
        void                ReportionImage(Vector2i start = Zero2i,
                                           Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void                ResizeImage(Vector2i resolution) override;
        void                ResetImage() override;
};

inline void TracerBase::AttachTracerCallbacks(TracerCallbacksI& tc)
{
    callbacks = &tc;
}