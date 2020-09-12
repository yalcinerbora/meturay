#pragma once

#include <cuda.h>

#include "RayLib/Vector.h"
#include "RayLib/SceneStructs.h"

#include "DeviceMemory.h"
#include "GPUMediumI.h"

class GPUMediumHomogenous : public GPUMediumI
{
    private:
        Vector3         sigmaA;
        Vector3         sigmaS;
        Vector3         sigmaT;
        float           ior;
        float           phase;

        uint32_t        id;

    public:
        // Constructors & Destructor
        __device__                  GPUMediumHomogenous(Vector3 sigmaA,
                                                      Vector3 sigmaS,
                                                      float ior, float phase,
                                                      uint32_t id);
        virtual                     ~GPUMediumHomogenous() = default;

       // Interface
       __device__ Vector3           SigmaA() const override;
       __device__ Vector3           SigmaS() const override;
       __device__ Vector3           SigmaT() const override;
       __device__ float             IOR() const override;
       __device__ float             Phase() const override;
       __device__ uint32_t          ID() const override;

       __device__ Vector3           Transmittance(float distance) const override;
};

class CPUMediumHomogenous : public CPUMediumGroupI
{
    public:
        static const char* TypeName() { return "Homogenous"; }

        static constexpr const char*    ABSORBTION = "absorption";
        static constexpr const char*    SCATTERING = "scattering";
        static constexpr const char*    IOR = "ior";
        static constexpr const char*    Phase = "phase";


    private:
        DeviceMemory                    memory;

        GPUMediumList                   gpuMediumList;

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
		TracerError					ConstructMediums(const CudaSystem&) override;
		uint32_t					MediumCount() const override;

		size_t						UsedGPUMemory() const override;
		size_t						UsedCPUMemory() const override;
};


__device__
inline GPUMediumHomogenous::GPUMediumHomogenous(Vector3 sigmaA,
                                                Vector3 sigmaS,
                                                float ior, float phase,
                                                uint32_t id)
    : sigmaA(sigmaA)
    , sigmaS(sigmaS)
    , sigmaT(sigmaA + sigmaS)
    , ior(ior)
    , phase(phase)
    , id(id)
{}

__device__ inline Vector3 GPUMediumHomogenous::SigmaA() const {return sigmaA;}
__device__ inline Vector3 GPUMediumHomogenous::SigmaS() const { return sigmaS; }
__device__ inline Vector3 GPUMediumHomogenous::SigmaT() const { return sigmaT; }
__device__ inline float GPUMediumHomogenous::IOR() const { return ior; }
__device__ inline float GPUMediumHomogenous::Phase() const { return phase; }
__device__ inline uint32_t GPUMediumHomogenous::ID() const { return id; }

__device__
inline Vector3 GPUMediumHomogenous::Transmittance(float distance) const
{
    constexpr Vector3 Zero = Zero3;
    if(sigmaT == Zero) return Vector3(1.0f);
    if(distance == INFINITY) return Zero;

    // Beer's Law
    Vector3 result = (-sigmaT) * distance;
    result[0] = expf(result[0]);
    result[1] = expf(result[1]);
    result[2] = expf(result[2]);

    //printf("%f, %f, %f --- %f, %f, %f --- %f, %f, %f -- %f\n", 
    //       result[0], result[1], result[2],
    //       result2[0], result2[1], result2[2],
    //       sigmaT[0], sigmaT[1], sigmaT[2],
    //       distance);

    return result;
}

inline const char* CPUMediumHomogenous::Type() const
{
    return TypeName();
}

inline const GPUMediumList& CPUMediumHomogenous::GPUMediums() const
{
    return gpuMediumList;
}

inline uint32_t CPUMediumHomogenous::MediumCount() const
{
    return static_cast<uint32_t>(gpuMediumList.size());
}

inline size_t CPUMediumHomogenous::UsedGPUMemory() const
{
    return memory.Size();
}

inline size_t CPUMediumHomogenous::UsedCPUMemory() const
{
    return 0;
}