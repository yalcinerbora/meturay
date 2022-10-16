#pragma once

#include "RayLib/CudaCheck.h"
#include "RayLib/Vector.h"
#include "RayLib/Constants.h"
#include "RayLib/HybridFunctions.h"

// TODO: Implement more filters later,
// Change its location etc.
template <int32_t N>
class StaticGaussianFilter1D
{
    static_assert(N % 2 == 1, "Filter kernel size must be odd");
    public:
    static constexpr Vector2i   KERNEL_RANGE = Vector2i(-N / 2, N / 2);
    private:
    float                       kernel[N];

    public:
    __host__                    StaticGaussianFilter1D(float alpha);
    __device__ float            operator()(int32_t i) const;
};

// Block Local "Texture" (2D array not CUDA texture or GPU texture)
// filtering class. This requires separable filtering functor
// Given array of local data (register space)
// this class does filtering over the data
// each thread has some part of the texture
// data per thread is strided meaning;
// data[0] is the "nth" element
// data[1] is the "TPB + nth" element etc.
template <class Filter,
          uint32_t TPB,
          uint32_t X, uint32_t Y>
class BlockTextureFilter2D
{
    private:
    static constexpr bool TPBCheck(uint32_t TPB, uint32_t X, uint32_t Y)
    {
        auto PIX_COUNT = (X * Y);
        if(TPB > PIX_COUNT) return TPB % PIX_COUNT == 0;
        if(TPB <= PIX_COUNT) return PIX_COUNT % TPB == 0;
        return false;
    }

    // No SFINAE, just static assert
    static_assert(TPBCheck(TPB, X, Y),
                  "TBP and (X * Y) must be divisible, (X*Y) / TBP or TBP / (X*Y)");
    static constexpr Vector2i   KERNEL_RANGE = Filter::KERNEL_RANGE;

    public:
    static constexpr uint32_t DATA_PER_THREAD = std::max(1u, (X * Y) / TPB);
    static constexpr uint32_t PIX_COUNT = (X * Y);

    struct TempStorage
    {
        float sTexture[Y][X];
    };

    private:
    TempStorage&    sMem;
    const Filter&   filterFunctor;
    const uint32_t  threadId;

    protected:
    public:
    // Constructors & Destructor
    __device__
                    BlockTextureFilter2D(TempStorage& storage,
                                         const Filter& filterFunctor);

    __device__
    void            operator()(float(&dataOut)[DATA_PER_THREAD],
                               const float(&data)[DATA_PER_THREAD]);

};

template <int32_t N>
__host__ inline
StaticGaussianFilter1D<N>::StaticGaussianFilter1D(float alpha)
{
    auto Gauss = [&](float t)
    {
        constexpr float W = 1.0f / MathConstants::SqrtPi / MathConstants::Sqrt2;
        return W / std::sqrt(alpha) * std::exp(-(t * t) * 0.5f / alpha);
    };
    // Generate weights
    for(int i = KERNEL_RANGE[0]; i <= KERNEL_RANGE[1]; i++)
    {
        kernel[i + N / 2] = Gauss(static_cast<float>(i));
    };

    // Normalize the Kernel
    float total = 0.0f;
    for(int i = 0; i < N; i++) { total += kernel[i]; }
    for(int i = 0; i < N; i++) { kernel[i] /= total; }
}

template <int32_t N>
__device__ inline
float StaticGaussianFilter1D<N>::operator()(int32_t i) const
{
    return kernel[i + (N / 2)];
}

template <class F,
          uint32_t TPB,
          uint32_t X, uint32_t Y>
__device__ inline
BlockTextureFilter2D<F, TPB, X, Y>::BlockTextureFilter2D(TempStorage& storage,
                                                         const F& filterFunctor)
    : sMem(storage)
    , filterFunctor(filterFunctor)
    , threadId(threadIdx.x)
{}

template <class F,
          uint32_t TPB,
          uint32_t X, uint32_t Y>
__device__ inline
void BlockTextureFilter2D<F, TPB, X, Y>::operator()(float(&dataOut)[DATA_PER_THREAD],
                                                    const float(&data)[DATA_PER_THREAD])
{
    auto LoadToSMem = [&](const float(&threadData)[DATA_PER_THREAD])
    {
        for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
        {
            // Find out the row/column indices
            uint32_t pixelId = (i * TPB) + threadId;
            uint32_t rowId = pixelId / X;
            uint32_t columnId = pixelId % X;
            // There may be more threads than pixels
            if(rowId < Y)
            {
                sMem.sTexture[rowId][columnId] = threadData[i];
            }
        }
    };

    // Directly load the data to the shared memory
    LoadToSMem(data); __syncthreads();

    // Do X pass
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        uint32_t pixelId = (i * TPB) + threadId;
        uint32_t rowId = pixelId / X;
        int32_t columnId = static_cast<int32_t>(pixelId % X);

        // Utilize dataOut as intermediate buffer
        dataOut[i] = 0.0f;
        for(int32_t j = KERNEL_RANGE[0]; j <= KERNEL_RANGE[1]; ++j)
        {
            int32_t col = HybridFuncs::Clamp<int32_t>(columnId + j, 0, X - 1);
            dataOut[i] += sMem.sTexture[rowId][col] * filterFunctor(j);
        }
    }
    __syncthreads();
    // Load data to shared memory again
    LoadToSMem(dataOut); __syncthreads();

    // Do Y pass
    for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
    {
        uint32_t pixelId = (i * TPB) + threadId;
        int32_t rowId = static_cast<int32_t>(pixelId / X);
        uint32_t columnId = pixelId % X;

        // After Y pass it is done, it is already on the
        // output buffer
        // TODO: Strided access check for bank conflicts
        dataOut[i] = 0.0f;
        for(int32_t j = KERNEL_RANGE[0]; j <= KERNEL_RANGE[1]; ++j)
        {
            int32_t row = HybridFuncs::Clamp<int32_t>(rowId + j, 0, Y - 1);
            dataOut[i] += sMem.sTexture[row][columnId] * filterFunctor(j);
        }
    }
    // All Done!
}


// Default Gauss Filter
// TODO: Change this later make the system more dynamic
using GaussFilter = StaticGaussianFilter1D<5>;