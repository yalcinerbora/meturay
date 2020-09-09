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

#include "RNGMemory.h"
#include "RayMemory.h"
#include "ImageMemory.h"

class GPUMedium;
class GPUTransformI;
class GPUSceneI;
class CudaSystem;

struct CPUTransform;
struct CPUMedium;
struct TracerError;

class GPUTracer : public GPUTracerI
{
    private:
        static constexpr const size_t       AlignByteCount = 128;

        // Max Bit Sizes for Efficient Sorting
        const Vector2i                              maxAccelBits;
        const Vector2i                              maxWorkBits;
        //
        const uint32_t                              maxHitSize;
        // Batches of Accelerator
        GPUBaseAcceleratorI&                        baseAccelerator;
        const AcceleratorBatchMap&                  accelBatches;
        // Batches of Material
        const std::map<NameGPUPair, GPUMatGPtr>&    materialGroups;

        const std::vector<CPUTransformGPtr>&        transforms;
        const std::vector<CPUMedium>&               mediums;
        std::vector<const GPUTransformI*>           hGPUTransforms;

        // GPU Memory
        DeviceMemory                                mediumAndTransformMemory;

    protected:
        // Cuda System For Kernel Calls
        const CudaSystem&                   cudaSystem;
        // GPU Memory
        RNGMemory                           rngMemory;
        RayMemory                           rayMemory;
        ImageMemory                         imgMemory;        
        const GPUTransformI* const*         dTransforms;        
        const GPUMedium*                    dMediums;
        // CPUMemory
        const GPUTransformI* const*         hTransforms;
        //
        TracerParameters                    params;
        //
        uint32_t                            currentRayCount;
        // Callbacks
        TracerCallbacksI*                   callbacks;
        bool                                crashed;
        // Current Work Partition
        RayPartitions<uint32_t>             workPartition;

        // Interface
        virtual void                        ResetHitMemory(uint32_t rayCount,
                                                           HitKey baseBoundMatKey);

        // Do a hit determination over current rays
        void                                HitAndPartitionRays();
        // Determine auxiliary size
        void                                WorkRays(const WorkBatchMap&,
                                                     const RayPartitions<uint32_t>& outPortions,
                                                     uint32_t totalRayOut,
                                                     HitKey baseBoundMatKey);

        // Internals
        template <class... Args>
        void                                SendLog(const char*, Args...);
        void                                SendError(TracerError e, bool isFatal);

        RayPartitions<uint32_t>             PartitionOutputRays(uint32_t& totalOutRay,
                                                                const WorkBatchMap&) const;

    public:
        // Constructors & Destructor
                                            GPUTracer(const CudaSystem&, 
                                                      const GPUSceneI&,
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