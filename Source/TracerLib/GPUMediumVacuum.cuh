#pragma once
#include <cuda.h>

#include "RayLib/Vector.h"
#include "RayLib/SceneStructs.h"

#include "AuxiliaryDataKernels.cuh"
#include "GPUMediumI.h"
#include "CudaConstants.h"
#include "DeviceMemory.h"

class GPUMediumVacuum : public GPUMediumI
{
    private:
    public:
        // Constructors & Destructor
                                    GPUMediumVacuum() = default;
        virtual                     ~GPUMediumVacuum() = default;

       // Interface
        __device__ Vector3           SigmaA() const override;
        __device__ Vector3           SigmaS() const override;
        __device__ Vector3           SigmaT() const override;
        __device__ float             IOR() const override;
        __device__ float             Phase() const override;
        __device__ uint32_t          ID() const override;

        __device__ Vector3           Transmittance(float distance) const override;
};

class CPUMediumVacuum : public CPUMediumGroupI
{
    public:
        static const char* TypeName() { return "Vacuum"; }
    private:
        DeviceMemory                memory;
        const GPUMediumVacuum*      dGPUMediums;
        GPUMediumList			    gpuMediumList;

    protected:
    public:
        // Constructors & Destructor
                                    CPUMediumVacuum() = default;
        virtual                     ~CPUMediumVacuum() = default;

        // Interface
        const char*                 Type() const override;
        const GPUMediumList&        GPUMediums() const override;
        SceneError					InitializeGroup(const NodeListing& transformNodes,
                                                    double time,
                                                    const std::string& scenePath) override;
        SceneError					ChangeTime(const NodeListing& transformNodes, double time,
                                               const std::string& scenePath) override;
        TracerError					ConstructMediums(const CudaSystem&) override;
        uint32_t					MediumCount() const override;

        size_t						UsedGPUMemory() const override;
        size_t						UsedCPUMemory() const override;
};

__device__ inline Vector3 GPUMediumVacuum::SigmaA() const { return 0.0f; }
__device__ inline Vector3 GPUMediumVacuum::SigmaS() const { return 0.0f; }
__device__ inline Vector3 GPUMediumVacuum::SigmaT() const { return 0.0f; }
__device__ inline float GPUMediumVacuum::IOR() const { return 1.0f; }
__device__ inline float GPUMediumVacuum::Phase() const { return 1.0f; }
__device__ inline uint32_t GPUMediumVacuum::ID() const { return 0; }

__device__
inline Vector3 GPUMediumVacuum::Transmittance(float distance) const
{
    return Vector3(1.0f);
}

inline const char* CPUMediumVacuum::Type() const
{
    return TypeName();
}

inline SceneError CPUMediumVacuum::InitializeGroup(const NodeListing& transformNodes,
                                                   double time,
                                                   const std::string& scenePath)
{
    DeviceMemory::EnlargeBuffer(memory, sizeof(GPUMediumVacuum));
    dGPUMediums = static_cast<GPUMediumVacuum*>(memory);
    return SceneError::OK;
}

inline SceneError CPUMediumVacuum::ChangeTime(const NodeListing& transformNodes, double time,
                                              const std::string& scenePath)
{
    return SceneError::OK;
}

inline TracerError CPUMediumVacuum::ConstructMediums(const CudaSystem& system)
{
    // Call allocation kernel
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    gpu.AsyncGridStrideKC_X(0, MediumCount(),
                            //
                            KCConstructGPUClass<GPUMediumVacuum>,
                            //
                            const_cast<GPUMediumVacuum*>(dGPUMediums),                            
                            MediumCount());

    gpu.WaitAllStreams();

    // Generate transform list
    for(uint32_t i = 0; i < MediumCount(); i++)
    {
        const auto* ptr = static_cast<const GPUMediumI*>(dGPUMediums + i);
        gpuMediumList.push_back(ptr);
    }
    return TracerError::OK;
}

inline const GPUMediumList& CPUMediumVacuum::GPUMediums() const
{
    return gpuMediumList;
}

inline uint32_t CPUMediumVacuum::MediumCount() const
{
    return 0;
}

inline size_t CPUMediumVacuum::UsedGPUMemory() const
{
    return 0;
}

inline size_t CPUMediumVacuum::UsedCPUMemory() const
{
    return memory.Size();
}