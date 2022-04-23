#pragma once

/**

Compile Time Cuda Constants

Thread per Block etc..

*/

#include <cuda.h>
#include <set>
#include <array>
#include <vector>
#include <algorithm>

#include "RayLib/Vector.h"
#include "RayLib/Error.h"

// Except first generation this did not change having this compile time constant is a bliss
static constexpr unsigned int WarpSize = 32;

// Thread Per Block Constants
static constexpr unsigned int StaticThreadPerBlock1D = 256;
static constexpr unsigned int StaticThreadPerBlock2D_X = 16;
static constexpr unsigned int StaticThreadPerBlock2D_Y = 16;
static constexpr Vector2ui StaticThreadPerBlock2D = Vector2ui(StaticThreadPerBlock2D_X,
                                                              StaticThreadPerBlock2D_Y);

// Cuda Kernel Optimization Hints
// Since we call all of the kernels in a static manner
// (in case of Block Size) hint the compiler
// using __launch_bounds__ expression
#define CUDA_LAUNCH_BOUNDS_1D __launch_bounds__(StaticThreadPerBlock1D)
#define CUDA_LAUNCH_BOUNDS_2D __launch_bounds__(StaticThreadPerBlock2D_X * StaticThreadPerBlock2D_Y);

struct CudaError : public ErrorI
{
    public:
        enum Type
        {
            OK,
            OLD_DRIVER,
            NO_DEVICE,
            // End
            END
        };

    private:
        Type        type;

    public:
        // Constructors & Destructor
                    CudaError(Type);
                    ~CudaError() = default;

         operator   Type() const;
         operator   std::string() const override;
};

inline CudaError::CudaError(CudaError::Type t)
    : type(t)
{}

inline CudaError::operator Type() const
{
    return type;
}

inline CudaError::operator std::string() const
{
    static constexpr char const* const ErrorStrings[] =
    {
        "OK",
        "Driver is not up-to-date",
        "No cuda capable device is found"
    };
    static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(CudaError::END),
                  "Enum and enum string list size mismatch.");

    return ErrorStrings[static_cast<int>(type)];
}

class CudaGPU
{
    public:
        enum GPUTier
        {
            GPU_UNSUPPORTED,
            GPU_KEPLER,
            GPU_MAXWELL,
            GPU_PASCAL,
            GPU_TURING_VOLTA,
            GPU_AMPERE
        };

        struct WorkGroup
        {
            static constexpr size_t                 MAX_STREAMS = 64;
            std::array<cudaEvent_t, MAX_STREAMS>    events;
            std::array<cudaStream_t, MAX_STREAMS>   works;
            cudaEvent_t                             mainEvent;
            mutable int                             currentIndex;
            int                                     totalStreams;

            // Constructors
                                            WorkGroup();
                                            WorkGroup(int deviceId, int streamCount);
                                            WorkGroup(const WorkGroup&) = delete;
                                            WorkGroup(WorkGroup&&);
            WorkGroup&                      operator=(const WorkGroup&) = delete;
            WorkGroup&                      operator=(WorkGroup&&);
                                            ~WorkGroup();

            cudaStream_t                    UseGroup() const;
            void                            WaitAllStreams() const;
            void                            WaitMainStream() const;
        };

    private:
        int                     deviceId;
        cudaDeviceProp          props;
        GPUTier                 tier;
        WorkGroup               workList;

        static GPUTier          DetermineGPUTier(cudaDeviceProp);
    public:
        uint32_t                DetermineGridStrideBlock(uint32_t sharedMemSize,
                                                         uint32_t threadCount,
                                                         size_t workCount,
                                                         void* func) const;

    protected:
    public:
        // Constrictors & Destructor
        explicit                CudaGPU(int deviceId);
                                CudaGPU(const CudaGPU&) = delete;
                                CudaGPU(CudaGPU&&) = default;
        CudaGPU&                operator=(const CudaGPU&) = delete;
        CudaGPU&                operator=(CudaGPU&&) = default;
                                ~CudaGPU() = default;
        //
        int                     DeviceId() const;
        std::string             Name() const;
        std::string             CC() const;
        double                  TotalMemoryMB() const;
        double                  TotalMemoryGB() const;
        GPUTier                 Tier() const;

        size_t                  TotalMemory() const;
        Vector2i                MaxTexture2DSize() const;

        uint32_t                SMCount() const;
        uint32_t                MaxActiveBlockPerSM(uint32_t threadsPerBlock = StaticThreadPerBlock1D) const;
        uint32_t                RecommendedBlockCountPerSM(const void* kernelPtr,
                                                           uint32_t threadsPerBlock = StaticThreadPerBlock1D,
                                                           uint32_t sharedMemSize = 0) const;
        cudaStream_t            DetermineStream(uint32_t requiredSMCount = 0) const;
        void                    WaitAllStreams() const;
        void                    WaitMainStream() const;

        bool                    operator<(const CudaGPU&) const;

        // Kernel Attribute Related Functions
        cudaFuncAttributes      GetKernelAttributes(const void* kernelPtr) const;
        bool                    SetKernelShMemSize(const void* kernelPtr, int sharedMemConfigSize) const;

        // Classic GPU Calls
        // Create just enough blocks according to work size
        template<class Function, class... Args>
        __host__ void           KC_X(uint32_t sharedMemSize,
                                     cudaStream_t stream,
                                     size_t workCount,
                                     //
                                     Function&& f, Args&&...) const;
        template<class Function, class... Args>
        __host__ void           KC_XY(uint32_t sharedMemSize,
                                      cudaStream_t stream,
                                      size_t workCount,
                                      //
                                      Function&& f, Args&&...) const;

        // Grid-Stride Kernels
        // Convenience Functions For Kernel Call
        // Simple partial/full GPU utilizing calls over a stream
        template<class Function, class... Args>
        __host__ void           GridStrideKC_X(uint32_t sharedMemSize,
                                               cudaStream_t stream,
                                               size_t workCount,
                                               //
                                               Function&&, Args&&...) const;

        template<class Function, class... Args>
        __host__ void           GridStrideKC_XY(uint32_t sharedMemSize,
                                                cudaStream_t stream,
                                                size_t workCount,
                                                //
                                                Function&&, Args&&...) const;

        // Smart GPU Calls
        // Automatic stream split
        // Only for grid-stride kernels
        // TODO:
        template<class Function, class... Args>
        __host__ void           AsyncGridStrideKC_X(uint32_t sharedMemSize,
                                                     size_t workCount,
                                                     //
                                                     Function&&, Args&&...) const;

        template<class Function, class... Args>
        __host__ void           AsyncGridStrideKC_XY(uint32_t sharedMemSize,
                                                     size_t workCount,
                                                     //
                                                     Function&&, Args&&...) const;

        // Exact Kernel Calls
        // You 1-1 specify block and grid dimensions
        template<class Function, class... Args>
        __host__ void           ExactKC_X(uint32_t sharedMemSize,
                                          cudaStream_t stream,
                                          uint32_t blockSize,
                                          uint32_t gridSize,
                                          //
                                          Function&&, Args&&...) const;
};

// Verbosity
using GPUList = std::set<CudaGPU>;

class CudaSystem
{
    private:
        GPUList                     systemGPUs;

    protected:
    public:
        // Constructors & Destructor
                                    CudaSystem() = default;
                                    CudaSystem(const CudaSystem&) = delete;

        CudaError                   Initialize();

        // Multi-Device Splittable Smart GPU Calls
        // Automatic device split and stream split on devices
        const std::vector<size_t>   GridStrideMultiGPUSplit(size_t workCount,
                                                            uint32_t threadCount,
                                                            uint32_t sharedMemSize,
                                                            void* f) const;

        // Misc
        const GPUList&              SystemGPUs() const;
        const CudaGPU&              BestGPU() const;

        size_t                      TotalMemory() const;

        // Device Synchronization
        void                        SyncAllGPUsMainStreamOnly() const;
        void                        SyncAllGPUs() const;

};

inline const GPUList& CudaSystem::SystemGPUs() const
{
    return systemGPUs;
}

inline const CudaGPU& CudaSystem::BestGPU() const
{
    // Return the largest memory GPU
    auto MemoryCompare = [](const CudaGPU& a, const CudaGPU& b)
    {
        return (a.TotalMemory() < b.TotalMemory());
    };
    auto element = std::max_element(systemGPUs.begin(), systemGPUs.end(), MemoryCompare);
    return *element;
}

inline void CudaSystem::SyncAllGPUsMainStreamOnly() const
{
    int currentDevice;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    for(const auto& gpu : systemGPUs)
    {
        gpu.WaitMainStream();
    }
    CUDA_CHECK(cudaSetDevice(currentDevice));
}

inline void CudaSystem::SyncAllGPUs() const
{
    int currentDevice;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    for(const auto& gpu : systemGPUs)
    {
        gpu.WaitAllStreams();
        break;
    }
    CUDA_CHECK(cudaSetDevice(currentDevice));
}

inline size_t CudaSystem::TotalMemory() const
{
    size_t memSize = 0;
    for(const auto& gpu : systemGPUs)
    {
        memSize += gpu.TotalMemory();
    }
    return memSize;
}