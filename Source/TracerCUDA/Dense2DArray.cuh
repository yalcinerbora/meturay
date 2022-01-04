#pragma once

#include "RayLib/Vector.h"
#include "DeviceMemory.h"

class Dense2DArrayGPU
{
    float*      gData;
    Vector2i    dataPerNode;

    __device__
    const float* Data(const uint32_t index);
};

class Dense2DArrayCPU
{
    private:
        DeviceMemory            memory;
        Dense2DArrayGPU         dArrayGPU;
    protected:
    public:

        const Dense2DArrayGPU  ArrayGPU() const;
};

inline const Dense2DArrayGPU Dense2DArrayCPU::ArrayGPU() const
{
    return dArrayGPU;
}

//namespace RLFunctions
//{
//    float PDF(const float* gData, const Vector2i& dim)
//    {
//        return 0;
//    }
//
//    Vector3f Sample(float& pdf, RNGeneratorGPUI& rng)
//    {
//        return Zero3f;
//    }
//}