#pragma once

#include "RayLib/Vector.h"
#include "DeviceMemory.h"
#include <random>
#include <cstdint>

template <uint32_t DIM>
class AngleHashGroupCPU;

template <uint32_t DIM>
class AngleHashGroupGPU
{
    private:
    uint32_t            hashCount;
    float*              gVectors;

    friend class        AngleHashGroupCPU<DIM>;

    public:
    __device__ void     Hash(uint32_t& gHashOut, const float value[DIM],
                             uint32_t hashIndex) const;
};

template <uint32_t DIM>
class AngleHashGroupCPU
{
    private:
        DeviceMemory                    memory;
        AngleHashGroupGPU<DIM>          hashGPU;

    protected:
    public:
        // Constructors & Destructor
                                        AngleHashGroupCPU(uint32_t hashCount,
                                                        uint32_t seed);
        const AngleHashGroupGPU<DIM>    HashGPU() const;

};

template <uint32_t DIM>
__device__ inline
void AngleHashGroupGPU<DIM>::Hash(uint32_t& gHashOut, const float value[DIM],
                                  uint32_t hashIndex) const
{
    float planeNormal[DIM];
    #pragma unroll
    for(int i = 0; i < DIM; i++)
        planeNormal[i] = gVectors[hashIndex * DIM + i];

    float hashVal = 0.0f;
    // Dot Product
    #pragma unroll
    for(int i = 0; i < DIM; i++)
    {
        hashVal += planeNormal[i] * value[i];
    }
    gHashOut = (hashVal > 0.0f) ? 1 : 0;
}

template <uint32_t DIM>
inline AngleHashGroupCPU<DIM>::AngleHashGroupCPU(uint32_t hashCount, uint32_t seed)
{
    float* dVectors;

    GPUMemFuncs::AllocateMultiData(std::tie(dVectors),
                                   memory,
                                   {hashCount * DIM});

    // Do the initialization on the CPU for now
    std::mt19937 rng(seed);
    std::normal_distribution<float> gaussian(0, 1.0);

    std::vector<float> hVectors(hashCount * DIM);

    // Select the vector component from normal dist
    for(float& vecComponent : hVectors)
    {
        vecComponent = gaussian(rng);
    }

    //for(uint32_t i = 0; i < hashCount; i++)
    //{
    //    float dot = 0;
    //    for(int j = 0; j < DIM; j++)
    //    {
    //        dot += hVectors[i * DIM + j] * hVectors[i * DIM + j];
    //    }
    //    dot = sqrt(dot);
    //    float lengthRecip = 1.0f / dot;
    //    for(int j = 0; j < DIM; j++)
    //    {
    //        hVectors[i * DIM + j] *= lengthRecip;
    //    }
    //}

    // Copy the GPU
    CUDA_CHECK(cudaMemcpy(dVectors, hVectors.data(),
                          sizeof(float) * DIM * hashCount,
                          cudaMemcpyHostToDevice));

    // Generate the struct
    hashGPU.hashCount = hashCount;
    hashGPU.gVectors = dVectors;
}

template <uint32_t DIM>
inline
const AngleHashGroupGPU<DIM> AngleHashGroupCPU<DIM>::HashGPU() const
{
    return hashGPU;
}