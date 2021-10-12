#pragma once

#include "RayLib/Vector.h"
#include "NodeListing.h"

class GPUMediumI;
class CudaSystem;

using GPUMediumList = std::vector<const GPUMediumI*>;

class GPUMediumI
{
    public:
        virtual             ~GPUMediumI() = default;
       // Interface
        __device__
        virtual Vector3     SigmaA() const = 0;
        __device__
        virtual Vector3     SigmaS() const = 0;
        __device__
        virtual Vector3     SigmaT() const = 0;
        __device__
        virtual float       IOR() const = 0;
        __device__
        virtual float       Phase() const = 0;
        __device__
        virtual uint32_t    GlobalIndex() const = 0;

        __device__
        virtual Vector3     Transmittance(float distance) const = 0;
};

class CPUMediumGroupI
{
    public:
        virtual                         ~CPUMediumGroupI() = default;

        // Interface
        virtual const char*             Type() const = 0;
        virtual const GPUMediumList&    GPUMediums() const = 0;
        virtual SceneError				InitializeGroup(const NodeListing& transformNodes,
                                                        double time, const std::string& scenePath) = 0;
        virtual SceneError				ChangeTime(const NodeListing& mediumNodes, double time,
                                                   const std::string& scenePath) = 0;
        virtual TracerError				ConstructMediums(const CudaSystem&,
                                                         uint32_t indexStartOffset) = 0;
        virtual uint32_t				MediumCount() const = 0;

        virtual size_t					UsedGPUMemory() const = 0;
        virtual size_t					UsedCPUMemory() const = 0;
};