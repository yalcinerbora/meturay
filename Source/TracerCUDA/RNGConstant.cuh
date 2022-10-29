#pragma once

#include "RNGenerator.h"
#include "RayLib/HybridFunctions.h"

// I mean, https://xkcd.com/221/ :)
// It is actually useful though (rarely)
//
// i.e. for camera ray generation you can supply this as RNG
// with value of zero and disable anti-aliasing.
class RNGConstantGPU : public RNGeneratorGPUI
{
    private:
    float                   value;

    protected:
    public:
        // Constructor
    __device__              RNGConstantGPU(float val = 4.0f) : value(val) {};
                            RNGConstantGPU(const RNGConstantGPU&) = delete;
    RNGConstantGPU&         operator=(const RNGConstantGPU&) = delete;
                            ~RNGConstantGPU() = default;

    __device__ float        Uniform() override {return value;}
    __device__ float        Uniform(float min, float max) { return HybridFuncs::Clamp(value, min, max); }
    __device__ Vector2f     Uniform2D() override { return Vector2f(value); }
    __device__ float        Normal() override { return value; }
    __device__ float        Normal(float mean, float stdDev) override { return value; }
};
