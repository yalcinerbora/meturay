#pragma once

#include "RayLib/Vector.h"
#include "DeviceMemory.h"
#include <random>

template <uint32_t DIM>
class L2HashGroupCPU;

template <uint32_t DIM>
class L2HashGroupGPU
{
    private:
    uint32_t            hashCount;
    float               width;
    float*              gOffsets;
    float*              gNormals;

    friend class        L2HashGroupCPU<DIM>;

    public:
    __device__ void     Hash(uint32_t& gHashOut, const float value[DIM],
                             uint32_t hashIndex) const;
};

template <uint32_t DIM>
class L2HashGroupCPU
{
    private:
        DeviceMemory            memory;
        L2HashGroupGPU<DIM>     hashGPU;

    protected:

    public:
        // Constructors & Destructor
                                    L2HashGroupCPU(uint32_t hashCount,
                                                   float width,
                                                   uint32_t seed);

        const L2HashGroupGPU<DIM>   HashGPU() const;

};

template <uint32_t DIM>
__device__ inline
void L2HashGroupGPU<DIM>::Hash(uint32_t& gHashOut, const float value[DIM],
                               uint32_t hashIndex) const
{
    float offset = gOffsets[hashIndex];
    float planeNormal[DIM];
    #pragma unroll
    for(int i = 0; i < DIM; i++)
        planeNormal[i] = gNormals[hashIndex * DIM + i];

    float hashVal = 0.0f;
    // Dot Product
    #pragma unroll
    for(int i = 0; i < DIM; i++)
    {
        hashVal += planeNormal[i] * value[i];
    }
    // Offset
    hashVal += offset;
    hashVal = floor(hashVal / width);
    hashVal = fabs(hashVal);
    gHashOut = static_cast<uint32_t>(hashVal);
}

template <uint32_t DIM>
inline
L2HashGroupCPU<DIM>::L2HashGroupCPU(uint32_t hashCount, float width, uint32_t seed)
{
    float* dOffsets;
    float* dNormals;
    GPUMemFuncs::AllocateMultiData(std::tie(dOffsets, dNormals),
                                   memory,
                                   {hashCount, hashCount * DIM});

    // Do the initialization on the CPU for now
    std::mt19937 rng(seed);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, width);

    std::vector<float> hOffsets(hashCount);
    std::vector<float> hNormals(hashCount * DIM);

    //float delta = 1.0f / static_cast<float>(hOffsets.size());
    for(int i = 0; i < hOffsets.size(); i++)
    {

        //hOffsets[i] = i * delta;
        hOffsets[i] = 0.0f;
    }

    // Select the vector component from normal dist
    for(float& normalComponent : hNormals)
    {
        normalComponent = gaussian(rng);
    }
    // Uniformly spread the Gaussian distributions
    //for(float& offset : hOffsets)
    //{
    //    offset = uniform(rng);
    //}

    // Copy the GPU
    CUDA_CHECK(cudaMemcpy(dOffsets, hOffsets.data(),
                          sizeof(float) * hashCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNormals, hNormals.data(),
                          sizeof(float) * DIM * hashCount,
                          cudaMemcpyHostToDevice));

    // Generate the struct
    hashGPU.hashCount = hashCount;
    hashGPU.width = width;
    hashGPU.gOffsets = dOffsets;
    hashGPU.gNormals = dNormals;
}

template <uint32_t DIM>
inline
const L2HashGroupGPU<DIM> L2HashGroupCPU<DIM>::HashGPU() const
{
    return hashGPU;
}