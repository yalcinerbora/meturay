#pragma once

#include "GPUDirectLightSamplerI.h"
#include "GPULightI.h"

class GPULightSamplerUniform : public GPUDirectLightSamplerI
{
    private:
        const GPULightI**   gLights;

        uint32_t            lightCount;

    protected:
    public:
        // Constructors & Destructor
        __device__          GPULightSamplerUniform(const GPULightI** gLights,
                                                   uint32_t lightCount);
                            ~GPULightSamplerUniform() = default;

        // Interface
        __device__
        bool                SampleLight(// Outputs
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
                                        RandomGPU& rng) const override;

        // Probablity density of sampling a particular light
        // (indicated by lightIndex) towards a position and direction
        // Conditional is the chance of sampling this direction on a selected light
        // Marginal one returns chance of sampling this particular light
        __device__
        void               Pdf(// Outputs
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
                               const Vector3& direction) const override;

        __device__
        void               Pdf(// Outputs
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
                               const QuatF& tbnRotation) const override;
};

__device__
inline GPULightSamplerUniform::GPULightSamplerUniform(const GPULightI** gLights,
                                                      uint32_t lightCount)
    : gLights(gLights)
    , lightCount(lightCount)
{}

// Interface
__device__
inline bool GPULightSamplerUniform::SampleLight(// Outputs
                                                // Work key if we "Hit" to the ray
                                                // (ray and light is not occluded)
                                                HitKey& key,
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
                                                RandomGPU& rng) const
{
    if(lightCount == 0) return false;

    // Randomly Select Light
    float r1 = GPUDistribution::Uniform<float>(rng);
    r1 *= static_cast<float>(lightCount);
    uint32_t index = static_cast<uint32_t>(r1);

    // Extremely rarely index becomes the light count
    // although GPUDistribution::Uniform should return [0, 1)
    // it still happens due to fp error i guess?
    // if it happens just return the last light on the list
    if(index == lightCount) index--;

    //printf("NEE Index %u total lights %u\n", index, lightCount);

    const GPULightI* light = gLights[index];
    light->Sample(lDistance, direction,
                  pdf, position, rng);
    // Incorporate the PDF of selecting that ligjt
    pdf *= (1.0f / static_cast<float>(lightCount));
    lightIndex = index;
    key = light->WorkKey();

    // Return infinite (or very large distance) for
    // primitive lights since those have to hit in order to function properly
    // For rest slightly nudge the distance for preventing
    // numerical unstability
    lDistance = (light->IsPrimitiveBackedLight())
                    ? FLT_MAX
                    : (lDistance + MathConstants::Epsilon);
    return true;
}

// Probablity density of sampling a particular light
// (indicated by lightIndex) towards a position and direction
__device__
inline void GPULightSamplerUniform::Pdf(// Outputs
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
                                        const Vector3& direction) const
{
    if(lightIndex >= lightCount)
    {
        marginal = 0.0f;
        conditional = 0.0f;
        return;
    }

    // Discrete sampling of such light (its uniform)
    float pdf = 1.0f / static_cast<float>(lightCount);

    // Probability of sampling such direction from the particular light
    float pdfLight = gLights[lightIndex]->Pdf(direction, position);

    marginal = pdf;
    conditional = pdfLight;
}

__device__
inline void GPULightSamplerUniform::Pdf(// Outputs
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
                                        const QuatF& tbnRotation) const
{
    if(lightIndex >= lightCount)
    {
        marginal = 0.0f;
        conditional = 0.0f;
        return;
    }

    // Discrete sampling of such light (its uniform)
    float pdf = 1.0f / static_cast<float>(lightCount);

    // Pdf of light when it is hit from this location
    float pdfLight = gLights[lightIndex]->Pdf(distance, hitPosition,
                                              direction, tbnRotation);
    // Probabilities are conditional thus multiply
    marginal = pdf;
    conditional = pdfLight;
}