#pragma once

#include "GPUEndpointI.h"
#include "GPUPrimitiveEmpty.h"

// Do not delete this file
// maybe these classes will be required to be implemented

struct UVSurface;
class GPULightI;

using GPULightList = std::vector<const GPULightI*>;

class GPULightI : public GPUEndpointI
{
    public:
        // Constructors & Destructor
        __device__                      GPULightI(uint16_t mediumIndex,
                                                  HitKey, const GPUTransformI&);
        virtual                         ~GPULightI() = default;

        // Interface
        virtual __device__ Vector3f     Emit(const Vector3& wo,
                                             const Vector3& pos,
                                             //
                                             const UVSurface&,
                                             float solidAngle = 0.0f) const = 0;
        virtual __device__ uint32_t     GlobalLightIndex() const = 0;
        virtual __device__ void         SetGlobalLightIndex(uint32_t) = 0;
        virtual __device__ bool         IsPrimitiveBackedLight() const = 0;

        virtual __device__ Vector3f     GeneratePhoton(// Output
                                                       RayReg& rayOut,
                                                       Vector3f& normal,
                                                       float& posPDF,
                                                       float& dirPDF,
                                                       // I-O
                                                       RNGeneratorGPUI&) const = 0;
};

class CPULightGroupI : public CPUEndpointGroupI
{
    public:
        virtual                         ~CPULightGroupI() = default;
        // Interface
        virtual const GPULightList&     GPULights() const = 0;
        virtual const CudaGPU&          GPU() const = 0;

        // TODO: maybe in future some lights does not support
        // dynamic inheritance (it should since currently lights are
        // dynamic interfaces)
        bool                            CanSupportDynamicInheritance() const;
};

__device__
inline GPULightI::GPULightI(uint16_t mediumIndex,
                            HitKey hk,
                            const GPUTransformI& gTrans)
    : GPUEndpointI(mediumIndex, hk, gTrans)
{}

inline bool CPULightGroupI::CanSupportDynamicInheritance() const
{
    return true;
}