#pragma once

// Cuda Event wrapper & Fence

#include <cuda.h>
#include <cuda_runtime.h>

class GPUFence
{
    private:
        cudaEvent_t     e;

    protected:
    public:
        // Constructors & Destructor
                    GPUFence(cudaStream_t s);
                    ~GPUFence();

        bool        CheckFence();
        void        WaitFence();
};

class GPUEvent
{};