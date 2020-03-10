#pragma once

#include "GPUEndpointI.cuh"
#include "RayLib/Camera.h"

class PinholeCamera final : public GPUEndpointI
{
    private:
        // Sample Ready Parameters
        // All of which is world space
        Vector3     position;
        Vector3     right;
        Vector3     up;
        Vector3     bottomLeft;         // Far plane bottom left
        Vector2     planeSize;          // Far plane size
        Vector2     nearFar;

    protected:
    public:
        // Constructors & Destructor
        __device__          PinholeCamera(const CPUCamera&);

        // Interface 
        __device__ void     Sample(// Output
                                   HitKey& materialKey,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RandomGPU&) const override;

        __device__ void     GenerateRay(// Output
                                        RayReg&,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU&) const override;
};

static constexpr size_t GPUCameraUnionSize = std::aligned_union<1, 
                                                                PinholeCamera>::alignment_value;