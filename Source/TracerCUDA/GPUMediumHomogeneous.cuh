#pragma once

#include <cuda.h>

#include "RayLib/Vector.h"
#include "RayLib/SceneStructs.h"

#include "DeviceMemory.h"
#include "GPUMediumI.h"
struct GPUHomogeneousMediumData
{
};

class GPUMediumHomogeneous final : public GPUMediumI
{
    public:
        struct Data
        {
            Vector3         sigmaA;
            Vector3         sigmaS;
            Vector3         sigmaT;
            float           ior;
            float           phase;
        };

    private:
        const Data&         data;
        uint32_t            index;

    public:
        // Constructors & Destructor
        __device__                  GPUMediumHomogeneous(const Data&, uint32_t index);
        virtual                     ~GPUMediumHomogeneous() = default;

       // Interface
       __device__ Vector3           SigmaA() const override;
       __device__ Vector3           SigmaS() const override;
       __device__ Vector3           SigmaT() const override;
       __device__ float             IOR() const override;
       __device__ float             Phase() const override;
       __device__ uint32_t          GlobalIndex() const override;

       __device__ Vector3           Transmittance(float distance) const override;
};

class CPUMediumHomogeneous final : public CPUMediumGroupI
{
    public:
        static const char* TypeName() { return "Homogeneous"; }

        static constexpr const char*    ABSORBTION  = "absorption";
        static constexpr const char*    SCATTERING  = "scattering";
        static constexpr const char*    IOR         = "ior";
        static constexpr const char*    PHASE       = "phase";

    private:
        DeviceMemory                        memory;
        const GPUMediumHomogeneous::Data*   dMediumData;
        const GPUMediumHomogeneous*         dGPUMediums;
        GPUMediumList                       gpuMediumList;
        uint32_t                            mediumCount;

    protected:
    public:
        // Interface
		const char*					Type() const override;
		const GPUMediumList&        GPUMediums() const override;
		SceneError					InitializeGroup(const NodeListing& mediumNodes,
													double time,
													const std::string& scenePath) override;
		SceneError					ChangeTime(const NodeListing& transformNodes, double time,
											   const std::string& scenePath) override;
		TracerError					ConstructMediums(const CudaSystem&,
                                                     uint32_t indexStartOffset) override;
		uint32_t					MediumCount() const override;

		size_t						UsedGPUMemory() const override;
		size_t						UsedCPUMemory() const override;
};

__device__
inline GPUMediumHomogeneous::GPUMediumHomogeneous(const GPUMediumHomogeneous::Data& d,
                                                 uint32_t index)
    : data(d)
    , index(index)
{}

__device__ inline Vector3 GPUMediumHomogeneous::SigmaA() const {return data.sigmaA;}
__device__ inline Vector3 GPUMediumHomogeneous::SigmaS() const { return data.sigmaS; }
__device__ inline Vector3 GPUMediumHomogeneous::SigmaT() const { return data.sigmaT; }
__device__ inline float GPUMediumHomogeneous::IOR() const { return data.ior; }
__device__ inline float GPUMediumHomogeneous::Phase() const { return data.phase; }
__device__ inline uint32_t GPUMediumHomogeneous::GlobalIndex() const { return index; }

__device__
inline Vector3 GPUMediumHomogeneous::Transmittance(float distance) const
{
    constexpr Vector3 Zero = Zero3;
    if(data.sigmaT == Zero) return Vector3(1.0f);
    if(distance == INFINITY) return Zero;

    // Beer's Law
    Vector3 result = (-data.sigmaT) * distance;
    result[0] = expf(result[0]);
    result[1] = expf(result[1]);
    result[2] = expf(result[2]);

    return result;
}

inline const char* CPUMediumHomogeneous::Type() const
{
    return TypeName();
}

inline const GPUMediumList& CPUMediumHomogeneous::GPUMediums() const
{
    return gpuMediumList;
}

inline uint32_t CPUMediumHomogeneous::MediumCount() const
{
    return mediumCount;
}

inline size_t CPUMediumHomogeneous::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPUMediumHomogeneous::UsedCPUMemory() const
{
    return 0;
}