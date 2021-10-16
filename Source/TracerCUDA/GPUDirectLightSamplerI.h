#pragma once

#include <cuda_runtime.h>
#include "RayLib/HitStructs.h"
#include "RayLib/Vector.h"
#include "RayLib/Quaternion.h"

#include "Random.cuh"

class GPUDirectLightSamplerI
{
    private:
    protected:
    public:
        // Interface
        __device__
        virtual bool SampleLight(// Outputs
                                 // Work key if we "Hit" to the ray (ray and light is not occluded)
                                 HitKey&,
                                 // Index of the light
                                 uint32_t& lightIndex,
                                 // Sampled Direction towards that light
                                 Vector3f& direction,
                                 // Distance between the lights
                                 float& lDistance,
                                 // Probability Density of such direction
                                 float& pdf,
                                 // Inputs
                                 // World location of the current shading point
                                 const Vector3& position,
                                 //
                                 RandomGPU& rng) const = 0;

        // Probablity density of sampling a particular light
        // (indicated by lightIndex) towards a position and direction
        // Conditional is the chance of sampling this direction on a selected light
        // Marginal one returns chance of sampling this particular light
        __device__
        virtual void    Pdf(// Outputs
                            // pdf of selecting this light
                            float& marginal,
                            // Selected Lights' pdf
                            float& conditional,
                            // Inputs
                            uint32_t lightIndex,
                            // From which world position and direction
                            // we want the pdf to be calculated
                            // (direction is towards the light)
                            const Vector3& position,
                            const Vector3& direction) const = 0;
        __device__
        virtual void    Pdf(// Outputs
                            // pdf of selecting this light
                            float& marginal,
                            // Selected Lights' pdf
                            float& conditional,
                            // Inputs
                            uint32_t lightIndex,
                            // From this hit location parameters
                            // direction calculate the pdf
                            float distance,
                            const Vector3& hitPosition,
                            const Vector3& direction,
                            const QuatF& tbnRotation) const = 0;
};
