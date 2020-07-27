#pragma once

// Cuda Event wrapper & Fence

#include <cuda.h>
#include <cuda_runtime.h>

class GPUFence
{
    private:
        cudaEvent_t     e = nullptr;

    protected:
    public:
        // Constructors & Destructor
                    GPUFence(cudaStream_t s);
                    GPUFence(const GPUFence&) = delete;
                    GPUFence(GPUFence&&);
                    GPUFence& operator=(const GPUFence&) = delete;
                    GPUFence& operator=(GPUFence&&);
                    ~GPUFence();

        bool        CheckFence();
        void        WaitFence();
};

inline GPUFence::GPUFence(cudaStream_t s)
{
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(e, s));
}

inline GPUFence::GPUFence(GPUFence&& other)
    : e(other.e)
{}

inline GPUFence& GPUFence::operator=(GPUFence&& other)
{
    assert(this != &other);
    if(e) CUDA_CHECK(cudaEventDestroy(e));

    e = other.e;
    other.e = nullptr;

    return *this;
}

inline GPUFence::~GPUFence()
{
    if(e) CUDA_CHECK(cudaEventDestroy(e));
}

inline bool GPUFence::CheckFence()
{
    cudaError err = cudaEventQuery(e);
    if(err == cudaSuccess) return true;
    else if(err == cudaErrorNotReady) return false;
    else
    {
        CUDA_CHECK_ERROR(err);
        return false;
    }
}

inline void GPUFence::WaitFence()
{
    CUDA_CHECK(cudaEventSynchronize(e));
}