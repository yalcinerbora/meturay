#include "WFPGTracer.h"
#include "RayTracer.hpp"

#include "RayLib/GPUSceneI.h"
#include "RayLib/Options.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/BitManipulation.h"
#include "RayLib/FileUtility.h"
#include "RayLib/VisorTransform.h"

#include "WFPGTracerWorks.cuh"
#include "GPULightSamplerUniform.cuh"
#include "GenerationKernels.cuh"
#include "GPUWork.cuh"
#include "GPUAcceleratorI.h"
#include "ParallelReduction.cuh"
#include "GPUReconFilterI.h"

#include <array>
#include <thread>

#ifdef MRAY_OPTIX
    #include "OctreeOptiX.h"
    #include "SVOOptiXRadianceBuffer.cuh"
    template <class T>
    using SegmentedField = SVOOptixRadianceBuffer::SegmentedField<T>;
#endif

// DEBUG
std::ostream& operator<<(std::ostream& stream, const RayAuxWFPG& v)
{
    stream << std::setw(0)
        << v.sampleIndex << ", "
        << "{" << v.radianceFactor[0]
        << "," << v.radianceFactor[1]
        << "," << v.radianceFactor[2] << "} "
        << v.endpointIndex << ", "
        << v.mediumIndex << " ";
    switch(v.type)
    {
        case RayType::CAMERA_RAY:
            stream << "CAMERA_RAY";
            break;
        case RayType::NEE_RAY:
            stream << "NEE_RAY";
            break;
        case RayType::SPECULAR_PATH_RAY:
            stream << "SPEC_PATH_RAY";
            break;
        case RayType::PATH_RAY:
            stream << "PATH_RAY";
    }
    stream << " {binId " << v.binId << "}";
    return stream;
}

template <int32_t THREAD_PER_BLOCK, int32_t X, int32_t Y>
__global__ __launch_bounds__(THREAD_PER_BLOCK)
void KCTraceSVOFromObjectCam(// Output
                             CamSampleGMem<Vector4f> gSamples,
                             // Input
                             const GPUCameraI* gCamera,
                             // Constants
                             const AnisoSVOctreeGPU svo,
                             WFPGRenderMode mode,
                             uint32_t maxQueryLevelOffset,

                             Vector2i totalPixelCount,
                             Vector2i totalSegments)
{
    TraceSVO<THREAD_PER_BLOCK, X, Y>(// Output
                                     gSamples,
                                     // Input
                                     gCamera,
                                     // Constants
                                     svo, mode,
                                     maxQueryLevelOffset,
                                     totalPixelCount,
                                     totalSegments);
}

template <int32_t THREAD_PER_BLOCK, int32_t X, int32_t Y>
__global__ __launch_bounds__(THREAD_PER_BLOCK)
void KCTraceSVOFromArrayCam(// Output
                            CamSampleGMem<Vector4f> gSamples,
                            // Input
                            const GPUCameraI** gCameras,
                            uint32_t cameraIndex,
                            // Constants
                            const AnisoSVOctreeGPU svo,
                            WFPGRenderMode mode,
                            uint32_t maxQueryLevelOffset,
                            // Camera region related
                            Vector2i totalPixelCount,
                            Vector2i totalSegments)
{
    TraceSVO<THREAD_PER_BLOCK, X, Y>(// Output
                                     gSamples,
                                     // Input
                                     gCameras[cameraIndex],
                                     // Constants
                                     svo, mode,
                                     maxQueryLevelOffset,
                                     totalPixelCount,
                                     totalSegments);
}

__global__
void KCCamPixelApertureFromArrayCam(float& gOutPixAperture,
                                    const GPUCameraI** gCameras,
                                    uint32_t cameraIndex,
                                    Vector2i resolution)
{
    if(threadIdx.x != 0) return;
    float solidAngle = GenSolidAngle(*(gCameras[cameraIndex]),
                                     resolution);
    gOutPixAperture = solidAngle;
}

__global__
void KCCamPixelApertureFromObjectCam(float& gOutPixAperture,
                                     const GPUCameraI* gCamera,
                                     Vector2i resolution)
{
    if(threadIdx.x != 0) return;
    float solidAngle = GenSolidAngle(*gCamera, resolution);
    gOutPixAperture = solidAngle;
}

// Currently These are compile time constants
// since most of the internal call rely on compile time constants
static constexpr uint32_t PG_KERNEL_TYPE_COUNT = 5;

constexpr float OctohedralConeAperture(uint32_t pixelCountX,
                                       uint32_t)
{
    constexpr float MAX_SOLID_ANGLE = 4.0f * MathConstants::Pi;
    float totalPixCount = (static_cast<float>(pixelCountX) *
                           static_cast<float>(pixelCountX));
    return MAX_SOLID_ANGLE / totalPixCount;
}

using PathGuideKernelFunction = void (*)(// Output
                                         RayAuxWFPG*,
                                         // I-O
                                         RNGeneratorGPUI**,
                                         // Input
                                         // Per-ray
                                         const RayId*,
                                         const GPUMetaSurfaceGeneratorGroup,
                                         // Per bin
                                         const uint32_t*,
                                         const uint32_t*,
                                         // Constants
                                         const GaussFilter,
                                         float coneAperture,
                                         const AnisoSVOctreeGPU,
                                         uint32_t,
                                         bool,
                                         float);

using PathGuideFilterKernelFunction = void (*)(// I-O
                                               SVOOptixRadianceBuffer::SegmentedField<float*>,
                                               // Constants
                                               const GaussFilter,
                                               uint32_t);

using PathGuideOptiXKernelFunction = void (*)(// Output
                                              RayAuxWFPG*,
                                              // I-O
                                              RNGeneratorGPUI**,
                                              // Input
                                              // Per-ray
                                              const RayId*,
                                              const GPUMetaSurfaceGeneratorGroup,
                                              // Per bin
                                              const uint32_t*,
                                              const uint32_t*,
                                              // Buffer
                                              SVOOptixRadianceBuffer::SegmentedField<const float*>,
                                              // Constants
                                              const AnisoSVOctreeGPU,
                                              uint32_t,
                                              bool,
                                              float);

using ParallaxAdjustKernelFunction = void(*)(// I-O
                                             RayAuxWFPG*,
                                             // Input
                                             // Per-ray
                                             const RayId*,
                                             // Per bin
                                             const uint32_t*,
                                             const uint32_t*,
                                             const Vector4f*,
                                             // MetaSurfaceGenerator
                                             const GPUMetaSurfaceGeneratorGroup,
                                             // Buffer
                                             SVOOptixRadianceBuffer::SegmentedField<const float*>,
                                             // Constants
                                             const AnisoSVOctreeGPU,
                                             float,
                                             uint32_t);

// 1st param is "Thread per block", 2nd and third params are X,Y resolution of the generated texture
using WFPGKernelParamType = std::tuple<uint32_t, uint32_t, uint32_t>;

static constexpr std::array<WFPGKernelParamType, PG_KERNEL_TYPE_COUNT> PG_KERNEL_PARAMS =
{
    // Kernel is passed the register limit of the device,
    // compiling using 100s of registers :(.
    // Reduce the block size for at least on debug mode
    std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 128, 128), // First bounce good approximation
    std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 64, 64),   // Second bounce as well
    std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 32, 32),   // Third bounce not so much
    std::make_tuple(256, 16, 16),                           // Fourth bounce not good
    std::make_tuple(128, 8, 8)                              // Fifth is bad
};

//static constexpr std::array<WFPGKernelParamType, PG_KERNEL_TYPE_COUNT> PG_KERNEL_PARAMS =
//{
//    std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 128, 128),
//    std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 128, 128),
//    std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 128, 128),
//    std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 128, 128),
//    std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 128, 128)
//
//    //std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 64, 64),
//    //std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 64, 64),
//    //std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 64, 64),
//    //std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 64, 64),
//    //std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 64, 64)
//
//    //std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 32, 32),
//    //std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 32, 32),
//    //std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 32, 32),
//    //std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 32, 32),
//    //std::make_tuple(METU_DEBUG_BOOL ? 256 : 512, 32, 32)
//
//    //std::make_tuple(256, 16, 16),
//    //std::make_tuple(256, 16, 16),
//    //std::make_tuple(256, 16, 16),
//    //std::make_tuple(256, 16, 16),
//    //std::make_tuple(256, 16, 16)
//
//    //std::make_tuple(256, 8, 8),
//    //std::make_tuple(256, 8, 8),
//    //std::make_tuple(256, 8, 8),
//    //std::make_tuple(256, 8, 8),
//    //std::make_tuple(256, 8, 8)
//};

static constexpr uint32_t KERNEL_TBP_MAX = std::get<0>(*std::max_element(PG_KERNEL_PARAMS.cbegin(),
                                                                         PG_KERNEL_PARAMS.cend(),
                                                                         [](const auto& t0, const auto& t1)
                                                                         {
                                                                             return std::get<0>(t0) < std::get<0>(t1);
                                                                         }));

static constexpr std::array<PathGuideKernelFunction, PG_KERNEL_TYPE_COUNT> PG_KERNELS =
{

    KCGenAndSampleDistribution<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[0]), std::get<1>(PG_KERNEL_PARAMS[0]), std::get<2>(PG_KERNEL_PARAMS[0])>,
    KCGenAndSampleDistribution<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[1]), std::get<1>(PG_KERNEL_PARAMS[1]), std::get<2>(PG_KERNEL_PARAMS[1])>,
    KCGenAndSampleDistribution<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[2]), std::get<1>(PG_KERNEL_PARAMS[2]), std::get<2>(PG_KERNEL_PARAMS[2])>,
    KCGenAndSampleDistribution<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[3]), std::get<1>(PG_KERNEL_PARAMS[3]), std::get<2>(PG_KERNEL_PARAMS[3])>,
    KCGenAndSampleDistribution<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[4]), std::get<1>(PG_KERNEL_PARAMS[4]), std::get<2>(PG_KERNEL_PARAMS[4])>
};

static constexpr std::array<PathGuideKernelFunction, PG_KERNEL_TYPE_COUNT> PG_PRODUCT_KERNELS =
{

    KCGenAndSampleDistributionProduct<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[0]), std::get<1>(PG_KERNEL_PARAMS[0]), std::get<2>(PG_KERNEL_PARAMS[0])>,
    KCGenAndSampleDistributionProduct<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[1]), std::get<1>(PG_KERNEL_PARAMS[1]), std::get<2>(PG_KERNEL_PARAMS[1])>,
    KCGenAndSampleDistributionProduct<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[2]), std::get<1>(PG_KERNEL_PARAMS[2]), std::get<2>(PG_KERNEL_PARAMS[2])>,
    KCGenAndSampleDistributionProduct<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[3]), std::get<1>(PG_KERNEL_PARAMS[3]), std::get<2>(PG_KERNEL_PARAMS[3])>,
    KCGenAndSampleDistributionProduct<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[4]), std::get<1>(PG_KERNEL_PARAMS[4]), std::get<2>(PG_KERNEL_PARAMS[4])>
};

static constexpr std::array<PathGuideOptiXKernelFunction, PG_KERNEL_TYPE_COUNT> PG_OPTIX_KERNELS =
{

    KCSampleDistributionOptiX<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[0]), std::get<1>(PG_KERNEL_PARAMS[0]), std::get<2>(PG_KERNEL_PARAMS[0])>,
    KCSampleDistributionOptiX<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[1]), std::get<1>(PG_KERNEL_PARAMS[1]), std::get<2>(PG_KERNEL_PARAMS[1])>,
    KCSampleDistributionOptiX<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[2]), std::get<1>(PG_KERNEL_PARAMS[2]), std::get<2>(PG_KERNEL_PARAMS[2])>,
    KCSampleDistributionOptiX<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[3]), std::get<1>(PG_KERNEL_PARAMS[3]), std::get<2>(PG_KERNEL_PARAMS[3])>,
    KCSampleDistributionOptiX<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[4]), std::get<1>(PG_KERNEL_PARAMS[4]), std::get<2>(PG_KERNEL_PARAMS[4])>
};

static constexpr std::array<PathGuideOptiXKernelFunction, PG_KERNEL_TYPE_COUNT> PG_OPTIX_PRODUCT_KERNELS =
{

    KCSampleDistributionProductOptiX<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[0]), std::get<1>(PG_KERNEL_PARAMS[0]), std::get<2>(PG_KERNEL_PARAMS[0])>,
    KCSampleDistributionProductOptiX<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[1]), std::get<1>(PG_KERNEL_PARAMS[1]), std::get<2>(PG_KERNEL_PARAMS[1])>,
    KCSampleDistributionProductOptiX<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[2]), std::get<1>(PG_KERNEL_PARAMS[2]), std::get<2>(PG_KERNEL_PARAMS[2])>,
    KCSampleDistributionProductOptiX<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[3]), std::get<1>(PG_KERNEL_PARAMS[3]), std::get<2>(PG_KERNEL_PARAMS[3])>,
    KCSampleDistributionProductOptiX<RNGIndependentGPU, std::get<0>(PG_KERNEL_PARAMS[4]), std::get<1>(PG_KERNEL_PARAMS[4]), std::get<2>(PG_KERNEL_PARAMS[4])>
};

static constexpr std::array<PathGuideFilterKernelFunction, PG_KERNEL_TYPE_COUNT> PG_GAUSS_FILTER_KERNELS =
{

    KCFilterRadianceFields<std::get<0>(PG_KERNEL_PARAMS[0]), std::get<1>(PG_KERNEL_PARAMS[0]), std::get<2>(PG_KERNEL_PARAMS[0])>,
    KCFilterRadianceFields<std::get<0>(PG_KERNEL_PARAMS[1]), std::get<1>(PG_KERNEL_PARAMS[1]), std::get<2>(PG_KERNEL_PARAMS[1])>,
    KCFilterRadianceFields<std::get<0>(PG_KERNEL_PARAMS[2]), std::get<1>(PG_KERNEL_PARAMS[2]), std::get<2>(PG_KERNEL_PARAMS[2])>,
    KCFilterRadianceFields<std::get<0>(PG_KERNEL_PARAMS[3]), std::get<1>(PG_KERNEL_PARAMS[3]), std::get<2>(PG_KERNEL_PARAMS[3])>,
    KCFilterRadianceFields<std::get<0>(PG_KERNEL_PARAMS[4]), std::get<1>(PG_KERNEL_PARAMS[4]), std::get<2>(PG_KERNEL_PARAMS[4])>
};

static constexpr std::array<ParallaxAdjustKernelFunction, PG_KERNEL_TYPE_COUNT> PG_PARALLAX_KERNELS =
{

    KCAdjustParallax<std::get<0>(PG_KERNEL_PARAMS[0]), std::get<1>(PG_KERNEL_PARAMS[0]), std::get<2>(PG_KERNEL_PARAMS[0])>,
    KCAdjustParallax<std::get<0>(PG_KERNEL_PARAMS[1]), std::get<1>(PG_KERNEL_PARAMS[1]), std::get<2>(PG_KERNEL_PARAMS[1])>,
    KCAdjustParallax<std::get<0>(PG_KERNEL_PARAMS[2]), std::get<1>(PG_KERNEL_PARAMS[2]), std::get<2>(PG_KERNEL_PARAMS[2])>,
    KCAdjustParallax<std::get<0>(PG_KERNEL_PARAMS[3]), std::get<1>(PG_KERNEL_PARAMS[3]), std::get<2>(PG_KERNEL_PARAMS[3])>,
    KCAdjustParallax<std::get<0>(PG_KERNEL_PARAMS[4]), std::get<1>(PG_KERNEL_PARAMS[4]), std::get<2>(PG_KERNEL_PARAMS[4])>
};

static constexpr std::array<uint32_t, PG_KERNEL_TYPE_COUNT> FILTER_KERNEL_SHMEM_SIZE =
{
    sizeof(KCFilterRadianceShMemOptiX<std::get<0>(PG_KERNEL_PARAMS[0]), std::get<1>(PG_KERNEL_PARAMS[0]), std::get<2>(PG_KERNEL_PARAMS[0])>),
    sizeof(KCFilterRadianceShMemOptiX<std::get<0>(PG_KERNEL_PARAMS[1]), std::get<1>(PG_KERNEL_PARAMS[1]), std::get<2>(PG_KERNEL_PARAMS[1])>),
    sizeof(KCFilterRadianceShMemOptiX<std::get<0>(PG_KERNEL_PARAMS[2]), std::get<1>(PG_KERNEL_PARAMS[2]), std::get<2>(PG_KERNEL_PARAMS[2])>),
    sizeof(KCFilterRadianceShMemOptiX<std::get<0>(PG_KERNEL_PARAMS[3]), std::get<1>(PG_KERNEL_PARAMS[3]), std::get<2>(PG_KERNEL_PARAMS[3])>),
    sizeof(KCFilterRadianceShMemOptiX<std::get<0>(PG_KERNEL_PARAMS[4]), std::get<1>(PG_KERNEL_PARAMS[4]), std::get<2>(PG_KERNEL_PARAMS[4])>)
};

static constexpr std::array<uint32_t, PG_KERNEL_TYPE_COUNT> PARALLAX_KERNEL_SHMEM_SIZE =
{
    sizeof(KCParallaxShMem<std::get<0>(PG_KERNEL_PARAMS[0]), std::get<1>(PG_KERNEL_PARAMS[0]), std::get<2>(PG_KERNEL_PARAMS[0])>),
    sizeof(KCParallaxShMem<std::get<0>(PG_KERNEL_PARAMS[1]), std::get<1>(PG_KERNEL_PARAMS[1]), std::get<2>(PG_KERNEL_PARAMS[1])>),
    sizeof(KCParallaxShMem<std::get<0>(PG_KERNEL_PARAMS[2]), std::get<1>(PG_KERNEL_PARAMS[2]), std::get<2>(PG_KERNEL_PARAMS[2])>),
    sizeof(KCParallaxShMem<std::get<0>(PG_KERNEL_PARAMS[3]), std::get<1>(PG_KERNEL_PARAMS[3]), std::get<2>(PG_KERNEL_PARAMS[3])>),
    sizeof(KCParallaxShMem<std::get<0>(PG_KERNEL_PARAMS[4]), std::get<1>(PG_KERNEL_PARAMS[4]), std::get<2>(PG_KERNEL_PARAMS[4])>)
};

static constexpr std::array<uint32_t, PG_KERNEL_TYPE_COUNT> PG_KERNEL_SHMEM_SIZE =
{
    sizeof(KCGenSampleShMem<std::get<0>(PG_KERNEL_PARAMS[0]), std::get<1>(PG_KERNEL_PARAMS[0]), std::get<2>(PG_KERNEL_PARAMS[0])>),
    sizeof(KCGenSampleShMem<std::get<0>(PG_KERNEL_PARAMS[1]), std::get<1>(PG_KERNEL_PARAMS[1]), std::get<2>(PG_KERNEL_PARAMS[1])>),
    sizeof(KCGenSampleShMem<std::get<0>(PG_KERNEL_PARAMS[2]), std::get<1>(PG_KERNEL_PARAMS[2]), std::get<2>(PG_KERNEL_PARAMS[2])>),
    sizeof(KCGenSampleShMem<std::get<0>(PG_KERNEL_PARAMS[3]), std::get<1>(PG_KERNEL_PARAMS[3]), std::get<2>(PG_KERNEL_PARAMS[3])>),
    sizeof(KCGenSampleShMem<std::get<0>(PG_KERNEL_PARAMS[4]), std::get<1>(PG_KERNEL_PARAMS[4]), std::get<2>(PG_KERNEL_PARAMS[4])>)
};

static constexpr std::array<float, PG_KERNEL_TYPE_COUNT> CONE_APERTURES =
{
    OctohedralConeAperture(std::get<1>(PG_KERNEL_PARAMS[0]), std::get<2>(PG_KERNEL_PARAMS[0])),
    OctohedralConeAperture(std::get<1>(PG_KERNEL_PARAMS[1]), std::get<2>(PG_KERNEL_PARAMS[1])),
    OctohedralConeAperture(std::get<1>(PG_KERNEL_PARAMS[2]), std::get<2>(PG_KERNEL_PARAMS[2])),
    OctohedralConeAperture(std::get<1>(PG_KERNEL_PARAMS[3]), std::get<2>(PG_KERNEL_PARAMS[3])),
    OctohedralConeAperture(std::get<1>(PG_KERNEL_PARAMS[4]), std::get<2>(PG_KERNEL_PARAMS[4]))
};

__global__
void KCNormalizeImage(// I-O
                      Vector4f* dPixels,
                      //
                      Vector4f& dMax,
                      Vector4f& dMin,
                      //
                      Vector2i imgRes)
{
    int totalPixSize = imgRes.Multiply();

    Vector4f rangeRecip = Vector4f(1.0f) / (dMax - dMin);

    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalPixSize;
        threadId += (blockDim.x * gridDim.x))
    {
        Vector4f pix = dPixels[threadId];
        pix = (pix - dMin) * rangeRecip;

        // Do not change the alpha
        pix[3] = dMax[3];

        dPixels[threadId] = pix;
    }
}

__global__
void KCSetCamPosToPathChain(// Output
                            WFPGPathNode* gPathNodes,
                            // Input
                            const RayGMem* gRays,
                            // Constants
                            uint32_t maxPathNodePerRay,
                            uint32_t totalNodeCount,
                            uint32_t rayCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < rayCount;
        threadId += (blockDim.x * gridDim.x))
    {
        RayReg ray = RayReg(gRays, threadId);

        const uint32_t pathStartIndex = threadId * maxPathNodePerRay;
        gPathNodes[pathStartIndex].worldPosition = ray.ray.getPosition();
    }
}

struct NodeIdFetchFunctor
{
    __device__ inline
    uint32_t operator()(const RayAuxWFPG& aux) const
    {
        return aux.binId;
    }
};

void WFPGTracer::ResizeAndInitPathMemory()
{
    size_t totalPathNodeCount = TotalPathNodeCount();
    //METU_LOG("Allocating WFPGTracer global path buffer: Size {:d} MiB",
    //         totalPathNodeCount * sizeof(WFPGPathNode) / 1024 / 1024);

    GPUMemFuncs::EnlargeBuffer(pathMemory, totalPathNodeCount * sizeof(WFPGPathNode));
    dPathNodes = static_cast<WFPGPathNode*>(pathMemory);

    // Initialize Paths
    const CudaGPU& bestGPU = cudaSystem.BestGPU();
    if(totalPathNodeCount > 0)
        bestGPU.KC_X(0, 0, totalPathNodeCount,
                     //
                     KCInitializeWFPGPaths,
                     //
                     dPathNodes,
                     static_cast<uint32_t>(totalPathNodeCount));

    // Allocate the initial camera position in path chain
    // Path chain does not store direction in order to calculate Wo
    // wee need it
    const RayGMem* dRays = rayCaster->RaysIn();
    uint32_t rayCount = rayCaster->CurrentRayCount();
    bestGPU.GridStrideKC_X(0, (cudaStream_t)0, rayCount,
                           //
                           KCSetCamPosToPathChain,
                           //
                           // Output
                           dPathNodes,
                           // Input
                           dRays,
                           MaximumPathNodePerPath(),
                           static_cast<uint32_t>(totalPathNodeCount),
                           rayCount);

    //Debug::DumpBatchedMemToFile("__PathNodes", dPathNodes,
    //                            MaximumPathNodePerPath(),
    //                            totalPathNodeCount);

}

uint32_t WFPGTracer::TotalPathNodeCount() const
{
    return (imgMemory.SegmentSize()[0] * imgMemory.SegmentSize()[1] *
            options.sampleCount * options.sampleCount) * MaximumPathNodePerPath();
}

uint32_t WFPGTracer::MaximumPathNodePerPath() const
{
    return (options.maximumDepth == 0) ? 0 : (options.maximumDepth + 1);
}

void WFPGTracer::AccumulateRayHitsToSVO()
{
    const CudaGPU& gpu = cudaSystem.BestGPU();

    const RayGMem* dRays = rayCaster->RaysIn();
    RayAuxWFPG* dRayAux = static_cast<RayAuxWFPG*>(*dAuxIn);
    uint32_t rayCount = rayCaster->CurrentRayCount();

    //
    Byte* dRLETempMem;
    uint32_t* dBinIdBuffer0 = nullptr;
    uint32_t* dBinIdBuffer1 = nullptr;
    uint32_t* dRLEOutCountBuffer = nullptr;
    uint32_t* dRLECountAmount = nullptr;
    cub::DoubleBuffer<uint32_t> sortBuffer(dBinIdBuffer0, dBinIdBuffer1);

    size_t rleTempMemSize;
    cub::DeviceRunLengthEncode::Encode(nullptr, rleTempMemSize,
                                       dBinIdBuffer0, dBinIdBuffer1,
                                       dRLEOutCountBuffer, dRLECountAmount,
                                       rayCount);

    size_t sortTempMemSize;
    cub::DeviceRadixSort::SortKeysDescending(nullptr, sortTempMemSize,
                                             sortBuffer, rayCount);

    // Sort TempMem may be amortized because of the RLE output count buffer
    // Check it
    // Worst case scenario every ray hit to a different SVO leaf
    // (highly unlikely but still we need to allocate for that)
    size_t rleOutBufferSize = rayCount * sizeof(uint32_t);
    if(sortTempMemSize > rleOutBufferSize + rleTempMemSize)
    {
        // Add the different to RLE temp mem
        // Don't forget to allocate contagiously these two buffers though
        rleTempMemSize += sortTempMemSize - (rleOutBufferSize + rleTempMemSize);
    }

    GPUMemFuncs::AllocateMultiData(std::tie(dBinIdBuffer0,
                                            dBinIdBuffer1,
                                            dRLEOutCountBuffer,
                                            dRLETempMem,
                                            dRLECountAmount),
                                   partitionMemory,
                                   {rayCount, rayCount, rayCount,
                                   rleTempMemSize, 1});
    // Actually populate the double buffer now
    sortBuffer = cub::DoubleBuffer<uint32_t>(dBinIdBuffer0, dBinIdBuffer1);


    // Init ray bins
    gpu.GridStrideKC_X(0, (cudaStream_t)0, rayCount,
                       //
                       KCInitializeSVOBins,
                       // Outputs
                       sortBuffer.Current(),
                       dRayAux,
                       // Input
                       dRays,
                       rayCaster->KeysOut(),
                       scene.BaseBoundaryMaterial(),
                       svo.TreeGPU(),
                       rayCount);


    // Now do the partitioning
    // Sort
    cub::DeviceRadixSort::SortKeysDescending(dRLEOutCountBuffer,
                                             sortTempMemSize,
                                             sortBuffer, rayCount);
    CUDA_KERNEL_CHECK();
    // Then RLE
    // Rename buffers for clarity
    uint32_t* dSortedBinIds = sortBuffer.Current();
    uint32_t* dRLEOutBinIds = sortBuffer.Alternate();
    cub::DeviceRunLengthEncode::Encode(dRLETempMem, rleTempMemSize,
                                       dSortedBinIds, dRLEOutBinIds,
                                       dRLEOutCountBuffer, dRLECountAmount,
                                       rayCount);
    CUDA_KERNEL_CHECK();

    uint32_t hRLECountAmount;
    CUDA_CHECK(cudaMemcpy(&hRLECountAmount, dRLECountAmount,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));

    //Debug::DumpMemToFile("dSortedBinIds", dSortedBinIds, rayCount, false, true);
    //Debug::DumpMemToFile("dRLEOutBinIds", dRLEOutBinIds, hRLECountAmount, false, true);
    //Debug::DumpMemToFile("dRLEOutCountBuffer", dRLEOutCountBuffer, hRLECountAmount);

    // Then write to the SVO leafs
    gpu.GridStrideKC_X(0, (cudaStream_t)0, hRLECountAmount,
                       //
                       KCWriteLeafRayAmounts,
                       // Outputs
                       svo.TreeGPU(),
                       // Inputs
                       dRLEOutBinIds,
                       dRLEOutCountBuffer,
                       // Constants
                       hRLECountAmount);
}

void WFPGTracer::GenerateGuidedDirections()
{
    // DEBUG TIMER
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const CudaGPU& gpu = cudaSystem.BestGPU();
    // Cluster the rays according to their svo location
    RayAuxWFPG* dRayAux = static_cast<RayAuxWFPG*>(*dAuxIn);
    uint32_t rayCount = rayCaster->CurrentRayCount();

    CUDA_CHECK(cudaEventRecord(start));

    // Zero out the ray counts from the previous iteration
    svo.ClearRayCounts(cudaSystem);

    // Write how many rays hit to a certain leaf
    // Store it on the SVO
    AccumulateRayHitsToSVO();

    // While iterating with values user may forget to change the minRayBinLevel
    // to a valid value clamp it to the ray level
    uint32_t validBinLevel = std::min(options.octreeLevel,
                                      options.minRayBinLevel);
    // Then call SVO to reduce the bins
    svo.CollapseRayCounts(validBinLevel,
                          options.binRayCount,
                          cudaSystem);

    // Then rays check if their initial node is reduced
    gpu.GridStrideKC_X(0, (cudaStream_t)0, rayCount,
                       //
                       KCCheckReducedSVOBins,
                       //
                       dRayAux,
                       svo.TreeGPU(),
                       rayCount);
    // Partition the generated rays wrt. to the SVO nodeId
    uint32_t hPartitionCount;
    uint32_t* dPartitionOffsets;
    uint32_t* dPartitionBinIds;
    DeviceMemory partitionMem;
    // Custom Ray Partition
    rayCaster->PartitionRaysWRTCustomData(hPartitionCount,
                                          partitionMem,
                                          dPartitionOffsets,
                                          dPartitionBinIds,
                                          dRayAux,
                                          NodeIdFetchFunctor(),
                                          rayCount,
                                          cudaSystem);

    // Get Meta Surface
    const RayGMem* dRaysIn = rayCaster->RaysIn();
    const HitKey* dWorkKeys = rayCaster->WorkKeys();
    const PrimitiveId* dPrimIds = rayCaster->PrimitiveIds();
    const TransformId* dTransformIds = rayCaster->TransformIds();
    const HitStructPtr dHitStructPtr = rayCaster->HitSturctPtr();
    const auto metaSurfGenerator = metaSurfHandler.GetMetaSurfaceGroup(dRaysIn,
                                                                       dWorkKeys,
                                                                       dPrimIds,
                                                                       dTransformIds,
                                                                       dHitStructPtr);

    // Select the kernel depending on the path depth
    uint32_t kernelIndex = std::min(currentDepth, PG_KERNEL_TYPE_COUNT - 1);
    //uint32_t kernelIndex = 0;

    // Iteration of OptiX kernels (due to shared memory limitation)
    uint32_t iterCount = 1;

    // Change code path (OptiX or CUDA)
    //CUDA_CHECK(cudaEventRecord(start));
    if(!options.optiXTrace)
    {
        // Select product or non-product versions of the kernels
        auto KERNEL_LIST = options.productPG ? PG_PRODUCT_KERNELS : PG_KERNELS;

        // Calculate Kernel Call parameters (block size etc.)
        auto KCGenAndSampleKernel = KERNEL_LIST[kernelIndex];
        float coneAperture = CONE_APERTURES[kernelIndex];
        uint32_t kernelShmemSize = PG_KERNEL_SHMEM_SIZE[kernelIndex];
        uint32_t kernelTPB = std::get<0>(PG_KERNEL_PARAMS[kernelIndex]);
        RNGeneratorGPUI** gpuGenerators = pgSampleRNG.GetGPUGenerators(gpu);
        CUDA_CHECK(cudaFuncSetAttribute(KCGenAndSampleKernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kernelShmemSize));

        //uint32_t activeBlockSize = gpu.DetermineGridStrideBlock(kernelShmemSize,
        //                                                        kernelTPB,
        //                                                        hPartitionCount * kernelTPB,
        //                                                        reinterpret_cast<const void*>(KCGenAndSampleKernel));

        // Actual Kernel Call
        gpu.ExactKC_X(kernelShmemSize, (cudaStream_t)0,
                      kernelTPB, pgKernelBlockCount,
                      //
                      KCGenAndSampleKernel,
                      // Output
                      dRayAux,
                      // I-O
                      gpuGenerators,
                      // Input
                      // Per-ray
                      rayCaster->SortedRayIds(),
                      metaSurfGenerator,
                      // Per bin
                      dPartitionOffsets,
                      dPartitionBinIds,
                      // Constants
                      rFieldGaussFilter,
                      coneAperture,
                      svo.TreeGPU(),
                      hPartitionCount,
                      options.purePG,
                      options.misRatio);
    }
    else
    {
        // In OptiX there is no shared memory thus we need to segregate the kernel
        // into two. In order to transfer data between kernels we need to allocate heap buffer.
        //
        // Partitions may not be processed using a single pass calculate pass count
        Vector2i fieldResolution = Vector2i(std::get<1>(PG_KERNEL_PARAMS[kernelIndex]),
                                            std::get<2>(PG_KERNEL_PARAMS[kernelIndex]));
        SegmentedField<float*> fieldBuffer = radianceBufferOptiX->Segment(fieldResolution);
        SegmentedField<const float*> fieldBufferConst = std::as_const(*radianceBufferOptiX.get()).Segment(fieldResolution);
        iterCount = (hPartitionCount + fieldBuffer.FieldCount() - 1) / fieldBuffer.FieldCount();

        // Determine the CUDA Kernels
        auto KERNEL_LIST = options.productPG ? PG_OPTIX_PRODUCT_KERNELS : PG_OPTIX_KERNELS;
        auto KCSampleKernel = KERNEL_LIST[kernelIndex];
        float coneAperture = CONE_APERTURES[kernelIndex];
        RNGeneratorGPUI** gpuGenerators = pgSampleRNG.GetGPUGenerators(gpu);

        // Allocate the OpitX required parameters
        Vector4f* dRadianceFieldRayOrigins;
        Vector2f* dProjectionJitters;
        GPUMemFuncs::AllocateMultiData(std::tie(dRadianceFieldRayOrigins,
                                                dProjectionJitters),
                                       *binInfoBufferOptiX,
                                       {hPartitionCount, hPartitionCount});
        // Also pre-generate the bin information
        // "Ray origin, tMin, and jitter" since we cant share
        // these using shared memory as well
        uint32_t blockCount = (hPartitionCount + KERNEL_TBP_MAX - 1) / KERNEL_TBP_MAX;
        blockCount = min(pgKernelBlockCount, blockCount);
        gpu.ExactKC_X(0, (cudaStream_t)0,
                      KERNEL_TBP_MAX, blockCount,
                      //
                      KCGenerateBinInfoOptiX<RNGIndependentGPU>,
                      // Output
                      dRadianceFieldRayOrigins,
                      dProjectionJitters,
                      // I-O
                      gpuGenerators,
                      // Input
                      // Per-Ray
                      rayCaster->SortedRayIds(),
                      metaSurfGenerator,
                      // Per-Bin
                      dPartitionOffsets,
                      dPartitionBinIds,
                      // Constants
                      svo.TreeGPU(),
                      hPartitionCount);

        //Debug::DumpMemToFile("dPartitionOffsets",
        //                     dPartitionOffsets,
        //                     hPartitionCount);
        //Debug::DumpMemToFile("dPartitionBinIds",
        //                     dPartitionBinIds,
        //                     hPartitionCount);
        //Debug::DumpMemToFile("dRadianceFieldRayOrigins",
        //                     dRadianceFieldRayOrigins,
        //                     hPartitionCount);
        //Debug::DumpMemToFile("dProjectionJitters",
        //                     dProjectionJitters,
        //                     hPartitionCount);

        // Since we call multiple kernels, it is bad to memcopy "params"
        // from host to device multiple times. Memcpy most of it
        // then only memcopy single word as offset on each operation
        coneCasterOptiX->CopyRadianceMapGenParams(dRadianceFieldRayOrigins,
                                                  dProjectionJitters,
                                                  fieldBuffer,
                                                  options.optiXUseSceneAcc,
                                                  coneAperture);
        // Now do the iteration
        uint32_t processedBins = 0;
        for(uint32_t i = 0; i < iterCount; i++)
        {
            uint32_t leftBins = (hPartitionCount - processedBins);
            uint32_t processedBinThisIter = min(fieldBuffer.FieldCount(), leftBins);
            uint32_t totalRayCount = processedBinThisIter * fieldBuffer.FieldDim().Multiply();

            // First call tracing
            coneCasterOptiX->RadianceMapGen(processedBins,
                                            totalRayCount);
            // Now call the CUDA
            // Offset the buffers
            uint32_t* dLocalPartitionBinIds = dPartitionBinIds + processedBins;
            uint32_t* dLocalPartitionOffsets = dPartitionOffsets + processedBins;
            Vector4f* dLocalFieldOrigins = dRadianceFieldRayOrigins + processedBins;

            uint32_t kernelShmemSize = PG_KERNEL_SHMEM_SIZE[kernelIndex];
            uint32_t kernelTPB = std::get<0>(PG_KERNEL_PARAMS[kernelIndex]);


            //Debug::Dump2DDataToImage("filterBefore",
            //                         fieldBufferConst.FieldRadianceArray(0),
            //                         Vector2i(std::get<1>(PG_KERNEL_PARAMS[kernelIndex]),
            //                                  std::get<2>(PG_KERNEL_PARAMS[kernelIndex])),
            //                                  processedBinThisIter);
            //Debug::Dump2DDataToImage("filterDistBefore",
            //                         fieldBufferConst.FieldDistanceArray(0),
            //                         Vector2i(std::get<1>(PG_KERNEL_PARAMS[kernelIndex]),
            //                                  std::get<2>(PG_KERNEL_PARAMS[kernelIndex])),
            //                         processedBinThisIter);

            // Call Filter Kernel
            uint32_t filterSharedMemSize = FILTER_KERNEL_SHMEM_SIZE[kernelIndex];
            auto KCFilterKernel = PG_GAUSS_FILTER_KERNELS[kernelIndex];

            CUDA_CHECK(cudaFuncSetAttribute(KCFilterKernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            filterSharedMemSize));

            gpu.ExactKC_X(filterSharedMemSize, (cudaStream_t)0,
                          kernelTPB, pgKernelBlockCount,
                          //
                          KCFilterKernel,
                          //
                          fieldBuffer,
                          rFieldGaussFilter,
                          processedBinThisIter);

            //Debug::Dump2DDataToImage("filterAfter",
            //                         fieldBufferConst.FieldRadianceArray(0),
            //                         Vector2i(std::get<1>(PG_KERNEL_PARAMS[kernelIndex]),
            //                                  std::get<2>(PG_KERNEL_PARAMS[kernelIndex])),
            //                                  processedBinThisIter);
            //Debug::Dump2DDataToImage("filterDistAfter",
            //                         fieldBufferConst.FieldDistanceArray(0),
            //                         Vector2i(std::get<1>(PG_KERNEL_PARAMS[kernelIndex]),
            //                                  std::get<2>(PG_KERNEL_PARAMS[kernelIndex])),
            //                         processedBinThisIter);

            // Call Sample Kernel
            CUDA_CHECK(cudaFuncSetAttribute(KCSampleKernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            kernelShmemSize));
            gpu.ExactKC_X(kernelShmemSize, (cudaStream_t)0,
                          kernelTPB, pgKernelBlockCount,
                          //
                          KCSampleKernel,
                          // Output
                          dRayAux,
                          // I-O
                          gpuGenerators,
                          // Input
                          // Per-ray
                          rayCaster->SortedRayIds(),
                          metaSurfGenerator,
                          // Per bin
                          dLocalPartitionOffsets,
                          dLocalPartitionBinIds,
                          // Buffer
                          fieldBufferConst,
                          // Constants
                          svo.TreeGPU(),
                          processedBinThisIter,
                          options.purePG,
                          options.misRatio);

            if(options.adjustParallax)
            {
                auto KCAdjustParallaxKernel = PG_PARALLAX_KERNELS[kernelIndex];
                auto paralaxKernelShmemSize = PARALLAX_KERNEL_SHMEM_SIZE[kernelIndex];
                CUDA_CHECK(cudaFuncSetAttribute(KCAdjustParallaxKernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                paralaxKernelShmemSize));
                gpu.ExactKC_X(paralaxKernelShmemSize, (cudaStream_t)0,
                              kernelTPB, pgKernelBlockCount,
                              //
                              KCAdjustParallaxKernel,
                              // I-O
                              dRayAux,
                              // Input
                              // Per-ray
                              rayCaster->SortedRayIds(),
                              // Per bin
                              dLocalPartitionOffsets,
                              dLocalPartitionBinIds,
                              dLocalFieldOrigins,
                              // MetaSurfaceGenerator
                              metaSurfGenerator,
                              // Buffer
                              fieldBufferConst,
                              // Constants
                              svo.TreeGPU(),
                              coneAperture,
                              processedBinThisIter);

            }
            processedBins += processedBinThisIter;
        }
        assert(processedBins == hPartitionCount);


    }
    CUDA_CHECK(cudaEventRecord(stop));

    // Only Consider useful rays for bins
    // Partition function sorts rays in descending order
    // first two offset values ("[0, n)") determine the invalid
    // ray range, subtract it from the ray count
    // (these include NEE rays, missed rays etc.)
    uint32_t invalidRayCount;
    CUDA_CHECK(cudaMemcpy(&invalidRayCount, dPartitionOffsets + 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    float milliseconds = 0;
    float avgRayPerBin = (static_cast<float>(rayCount - invalidRayCount) /
                          static_cast<float>(hPartitionCount - 1));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    //METU_LOG("Depth {:d} -> PartitionCount {:d}, AvgRayPerBin {:f}, KernelTime {:f}ms",
    //         currentDepth, hPartitionCount, avgRayPerBin, milliseconds);
    METU_LOG("{:d}; {:d}; {:f}; {:f}; i:{}",
             currentDepth, hPartitionCount,
             avgRayPerBin, milliseconds,
             iterCount);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void WFPGTracer::LaunchDebugConeTraceKernel()
{
    // TIMER DEBUG
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    DeviceMemory mem;
    float* dPixelAperture = nullptr;
    if(options.optiXTrace)
    {
        mem = DeviceMemory(sizeof(float));
        dPixelAperture = static_cast<float*>(mem);
    }

    // Calculate segment sizes etc
    static constexpr Vector2i REGION_SIZE = Vector2i(32, 32);
    // On debug, register count is too high we reduce the thread per block instead
    static constexpr int32_t THREAD_COUNT = METU_DEBUG_BOOL ? 256 : 512;
    Vector2i totalPixelCount = imgMemory.SegmentSize();
    Vector2i totalSegments = (totalPixelCount + (REGION_SIZE - Vector2i(1))) / REGION_SIZE;
    Vector2i extras = totalPixelCount % REGION_SIZE;
    totalPixelCount += extras;
    // Change the sample to nearest multiple the region size
    sampleMemory.Resize(imgMemory.Format(), totalPixelCount.Multiply());

    //
    uint32_t sharedMemSize = sizeof(KCTraceSVOSharedMem<THREAD_COUNT, REGION_SIZE[0], REGION_SIZE[1]>);

    auto Kernel0 = KCTraceSVOFromArrayCam<THREAD_COUNT, REGION_SIZE[0], REGION_SIZE[1]>;
    auto Kernel1 = KCTraceSVOFromObjectCam<THREAD_COUNT, REGION_SIZE[0], REGION_SIZE[1]>;
    CUDA_CHECK(cudaFuncSetAttribute(Kernel0,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    sharedMemSize));
    CUDA_CHECK(cudaFuncSetAttribute(Kernel1,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    sharedMemSize));

    bool enableAA = (options.renderMode == WFPGRenderMode::RENDER ||
                     options.renderMode == WFPGRenderMode::SVO_RADIANCE);

    // Generate rays appropriate to the camera type
    switch(currentCamera.type)
    {
        case SCENE_CAMERA:
        {
            const auto& gpu = cudaSystem.BestGPU();
            if(!options.optiXTrace)
            {
                CUDA_CHECK(cudaEventRecord(start));
                gpu.ExactKC_X(sharedMemSize, (cudaStream_t)0,
                              THREAD_COUNT, gpu.SMCount() * 2,
                              //
                              KCTraceSVOFromArrayCam<THREAD_COUNT, REGION_SIZE[0], REGION_SIZE[1]>,
                              // Output
                              sampleMemory.GMem<Vector4f>(),
                              // Input
                              dCameras,
                              currentCamera.nonTransformedCamIndex,
                              // Constants
                              svo.TreeGPU(),
                              options.renderMode,
                              options.svoRenderLevel,
                              // Camera region related
                              totalPixelCount,
                              totalSegments);
                CUDA_CHECK(cudaEventRecord(stop));

            }
            else
            {
                GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU, Vector4f>
                (
                    currentCamera.nonTransformedCamIndex,
                    options.sampleCount,
                    RayAuxInitWFPG(InitialWFPGAux,
                                   options.sampleCount *
                                   options.sampleCount),
                    true,
                    enableAA
                );

                gpu.ExactKC_X(0, (cudaStream_t)0, 1, 1,
                              //
                              KCCamPixelApertureFromArrayCam,
                              //
                              *dPixelAperture,
                              dCameras,
                              currentCamera.nonTransformedCamIndex,
                              imgMemory.SegmentSize());
            }
            break;
        }
        case CUSTOM_CAMERA:
        {
            const auto& gpu = cudaSystem.BestGPU();
            if(!options.optiXTrace)
            {
                // Now we can call the kernel
                CUDA_CHECK(cudaEventRecord(start));
                gpu.ExactKC_X(sharedMemSize, (cudaStream_t)0,
                              THREAD_COUNT, gpu.SMCount() * 2,
                              //
                              KCTraceSVOFromObjectCam<THREAD_COUNT, REGION_SIZE[0], REGION_SIZE[1]>,
                              // Output
                              sampleMemory.GMem<Vector4f>(),
                              // Input
                              currentCamera.dCustomCamera,
                              // Constants
                              svo.TreeGPU(),
                              options.renderMode,
                              options.svoRenderLevel,
                              // Camera region related
                              totalPixelCount,
                              totalSegments);
                CUDA_CHECK(cudaEventRecord(stop));
            }
            else
            {
                GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU, Vector4f>
                (
                    *currentCamera.dCustomCamera,
                    options.sampleCount,
                    RayAuxInitWFPG(InitialWFPGAux,
                                   options.sampleCount *
                                   options.sampleCount),
                    true,
                    enableAA
                );

                gpu.ExactKC_X(0, (cudaStream_t)0, 1, 1,
                              //
                              KCCamPixelApertureFromObjectCam,
                              //
                              *dPixelAperture,
                              currentCamera.dCustomCamera,
                              imgMemory.SegmentSize());
            }
            break;
        }
        case TRANSFORMED_SCENE_CAMERA:
        {
            uint32_t camIndex = currentCamera.transformedSceneCam.cameraIndex;
            VisorTransform t = currentCamera.transformedSceneCam.transform;
            const GPUCameraI* dCamera = GenerateCameraWithTransform(t, camIndex);
            const auto& gpu = cudaSystem.BestGPU();
            if(!options.optiXTrace)
            {
                // Now we can call the kernel
                CUDA_CHECK(cudaEventRecord(start));
                gpu.ExactKC_X(sharedMemSize, (cudaStream_t)0,
                              THREAD_COUNT, gpu.SMCount() * 2,
                              //
                              KCTraceSVOFromObjectCam<THREAD_COUNT, REGION_SIZE[0], REGION_SIZE[1]>,
                              // Output
                              sampleMemory.GMem<Vector4f>(),
                              // Input
                              dCamera,
                              // Constants
                              svo.TreeGPU(),
                              options.renderMode,
                              options.svoRenderLevel,
                              // Camera region related
                              totalPixelCount,
                              totalSegments);
                CUDA_CHECK(cudaEventRecord(stop));
            }
            else
            {
                GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU, Vector4f>
                (
                    t, camIndex, options.sampleCount,
                    RayAuxInitWFPG(InitialWFPGAux,
                                   options.sampleCount *
                                   options.sampleCount),
                    true,
                    enableAA
                );

                gpu.ExactKC_X(0, (cudaStream_t)0, 1, 1,
                              //
                              KCCamPixelApertureFromObjectCam,
                              //
                              *dPixelAperture,
                              dCamera,
                              imgMemory.SegmentSize());
            }
            break;
        }
        default: { crashed = true; break; }
    }

    if(options.optiXTrace)
    {
        float pixelAperture;
        CUDA_CHECK(cudaMemcpy(&pixelAperture, dPixelAperture,
                              sizeof(float), cudaMemcpyDeviceToHost));

        const RayGMem* dRays = rayCaster->RaysIn();
        RayAuxWFPG* dRayAux = static_cast<RayAuxWFPG*>(*dAuxIn);
        uint32_t rayCount = rayCaster->CurrentRayCount();

        CUDA_CHECK(cudaEventRecord(start));
        coneCasterOptiX->ConeTraceFromCamera(sampleMemory.GMem<Vector4f>(),
                                             //
                                             dRays,
                                             dRayAux,
                                             options.renderMode,
                                             options.svoRenderLevel,
                                             options.optiXUseSceneAcc,
                                             pixelAperture,
                                             rayCount);
        CUDA_CHECK(cudaEventRecord(stop));
    }


    float milliseconds = 0;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    METU_LOG("{}", milliseconds);
}

WFPGTracer::WFPGTracer(const CudaSystem& s,
                       const GPUSceneI& scene,
                       const TracerParameters& p)
    : RayTracer(s, scene, p)
    , currentDepth(0)
    , rFieldGaussFilter(options.rFieldGaussAlpha)
{
    // Append Work Types for generation
    boundaryWorkPool.AppendGenerators(WFPGBoundaryWorkerList{});
    pathWorkPool.AppendGenerators(WFPGPathWorkerList{});

    debugBoundaryWorkPool.AppendGenerators(WFPGDebugBoundaryWorkerList{});
    debugPathWorkPool.AppendGenerators(WFPGDebugPathWorkerList{});
}

TracerError WFPGTracer::Initialize()
{
    rFieldGaussFilter = GaussFilter(options.rFieldGaussAlpha);

    iterationCount = 0;
    treeDumpCount = 0;

    TracerError err = TracerError::OK;
    if((err = RayTracer::Initialize()) != TracerError::OK)
        return err;

    // Generate Light Sampler (for NEE)
    if((err = LightSamplerCommon::ConstructLightSampler(lightSamplerMemory,
                                                        dLightSampler,
                                                        options.lightSamplerType,
                                                        dLights,
                                                        lightCount,
                                                        cudaSystem)) != TracerError::OK)
        return err;

    // Generate your work list
    const auto& infoList = scene.WorkBatchInfo();
    for(const auto& wInfo : infoList)
    {
        WorkBatchArray workBatchList;
        uint32_t batchId = std::get<0>(wInfo);
        const GPUPrimitiveGroupI& pg = *std::get<1>(wInfo);
        const GPUMaterialGroupI& mg = *std::get<2>(wInfo);

        // Generic Path work
        GPUWorkBatchI* batch = nullptr;
        if(options.renderMode == WFPGRenderMode::SVO_INITIAL_HIT_QUERY)
        {
            WorkPool<>& wp = debugPathWorkPool;
            if((err = wp.GenerateWorkBatch(batch, mg, pg,
                                            dTransforms)) != TracerError::OK)
                return err;
        }
        else
        {
            WorkPool<bool, bool>& wpCombo = pathWorkPool;
            if((err = wpCombo.GenerateWorkBatch(batch, mg, pg,
                                                dTransforms,
                                                options.nextEventEstimation,
                                                options.directLightMIS)) != TracerError::OK)
                return err;
        }
        workBatchList.push_back(batch);
        workMap.emplace(batchId, workBatchList);
    }

    const auto& boundaryInfoList = scene.BoundarWorkBatchInfo();
    for(const auto& wInfo : boundaryInfoList)
    {
        uint32_t batchId = std::get<0>(wInfo);
        EndpointType et = std::get<1>(wInfo);
        const CPUEndpointGroupI& eg = *std::get<2>(wInfo);

        // Skip the camera types
        if(et == EndpointType::CAMERA) continue;

        WorkBatchArray workBatchList;
        GPUWorkBatchI* batch = nullptr;
        if(options.renderMode == WFPGRenderMode::SVO_INITIAL_HIT_QUERY)
        {
            BoundaryWorkPool<>& wp = debugBoundaryWorkPool;
            if((err = wp.GenerateWorkBatch(batch, eg,
                                           dTransforms)) != TracerError::OK)
                return err;
        }
        else
        {
            BoundaryWorkPool<bool, bool>& wp = boundaryWorkPool;
            if((err = wp.GenerateWorkBatch(batch, eg, dTransforms,
                                           options.nextEventEstimation,
                                           options.directLightMIS)) != TracerError::OK)
                return err;
        }
        workBatchList.push_back(batch);
        workMap.emplace(batchId, workBatchList);
    }

    // Init SVO
    if(options.svoInitPath.empty())
    {
        if((err = svo.Constrcut(scene.BaseAccelerator()->SceneExtents(),
                                (1 << options.octreeLevel),
                                scene.AcceleratorBatchMappings(),
                                dLights, lightCount,
                                scene.BaseBoundaryMaterial(),
                                options.omitLightRadiance,
                                cudaSystem)) != TracerError::OK)
            return err;
    }
    else
    {
        std::vector<Byte> data;
        Utility::DevourFileToStdVector(data, options.svoInitPath);
        if((err = svo.Constrcut(data,
                                dLights, lightCount,
                                scene.BaseBoundaryMaterial(),
                                cudaSystem)) != TracerError::OK)
            return err;
    }


    if(options.optiXTrace)
    {
        #ifdef MRAY_OPTIX
        if(this->optiXScene)
        {
            RayCasterI* rc = rayCaster.get();
            RayCasterOptiX* r = dynamic_cast<RayCasterOptiX*>(rc);
            const GPUBaseAcceleratorI& baseAccelOptiX = *scene.BaseAccelerator().get();
            coneCasterOptiX = std::make_unique<SVOOptixConeCaster>(r->GetOptiXSystem(),
                                                                   baseAccelOptiX,
                                                                   svo);
            coneCasterOptiX->GenerateSVOTraversable();
            size_t bufferSize = options.optiXBufferSize * 1024 * 1024 / sizeof(float) / 2;
            radianceBufferOptiX = std::make_unique<SVOOptixRadianceBuffer>(static_cast<uint32_t>(bufferSize));
            binInfoBufferOptiX = std::make_unique<DeviceMemory>();
        }
        else
            return TracerError(TracerError::TRACER_INTERNAL_ERROR,
                                "(WFPG) Enabling OptiX requires scene accelerators to be OptiX!");
        #else

        return TracerError(TracerError::TRACER_INTERNAL_ERROR,
                           "(WFPG) Executable does not support OptiX!");
        #endif
    }

    // Generate a Sampler for the
    // Path Guide Sampling (Conservatively generate maximum amount of RNGs)
    const auto& gpu = cudaSystem.BestGPU();
    uint32_t rngCount = (gpu.MaxActiveBlockPerSM(KERNEL_TBP_MAX) *
                         gpu.SMCount() * KERNEL_TBP_MAX);
    pgKernelBlockCount = rngCount / KERNEL_TBP_MAX;
    pgSampleRNG = RNGIndependentCPU(params.seed, gpu, rngCount);

    // Initialize The Meta Surface for Product Path Guiding
    if((err = metaSurfHandler.Initialize(scene, workMap)) != TracerError::OK)
        return err;

    return TracerError::OK;
}

TracerError WFPGTracer::SetOptions(const OptionsI& opts)
{
    TracerError err = TracerError::OK;
    if((err = opts.GetUInt(options.maximumDepth, MAX_DEPTH_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetInt(options.sampleCount, SAMPLE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.rrStart, RR_START_NAME)) != TracerError::OK)
        return err;

    std::string lightSamplerTypeString;
    if((err = opts.GetString(lightSamplerTypeString, LIGHT_SAMPLER_TYPE_NAME)) != TracerError::OK)
        return err;
    if((err = LightSamplerCommon::StringToLightSamplerType(options.lightSamplerType, lightSamplerTypeString)) != TracerError::OK)
        return err;

    if((err = opts.GetBool(options.nextEventEstimation, NEE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.directLightMIS, DIRECT_LIGHT_MIS_NAME)) != TracerError::OK)
        return err;
    // Method Related
    if((err = opts.GetUInt(options.octreeLevel, OCTREE_LEVEL_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.minRayBinLevel, RAY_BIN_MIN_LEVEL_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.binRayCount, BIN_RAY_COUNT_NAME)) != TracerError::OK)
        return err;

    std::string renderModeString;
    if((err = opts.GetString(renderModeString, RENDER_MODE_NAME)) != TracerError::OK)
        return err;
    if((err = StringToWFPGRenderMode(options.renderMode, renderModeString)) != TracerError::OK)
        return err;

    if((err = opts.GetUInt(options.svoRadRenderIter, SVO_DEBUG_ITER_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.svoRenderLevel, RENDER_LEVEL_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetString(options.svoInitPath, SVO_INIT_PATH_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetFloat(options.rFieldGaussAlpha, R_FIELD_GAUSS_ALPHA_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.skipPG, SKIP_PG_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.purePG, PURE_PG_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetFloat(options.misRatio, MIS_RATIO_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.productPG, PRODUCT_PG_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.optiXTrace, OPTIX_TRACE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.optiXUseSceneAcc, OPTIX_USE_SCENE_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.optiXBufferSize, OPTIX_BUFFER_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.omitLightRadiance, OMIT_LIGHT_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetBool(options.adjustParallax, PARALLAX_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.guideBounceCount, GUIDE_BOUNCE_COUNT_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetBool(options.doWeightedCombine, WEIGHTED_COMBINE_NAME)) != TracerError::OK)
        return err;

    if((err = opts.GetBool(options.pgDumpDebugData, PG_DUMP_DEBUG_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetUInt(options.pgDumpInterval, PG_DUMP_INTERVAL_NAME)) != TracerError::OK)
        return err;
    if((err = opts.GetString(options.pgDumpDebugName, PG_DUMP_PATH_NAME)) != TracerError::OK)
        return err;

    return TracerError::OK;
}

void WFPGTracer::AskOptions()
{
    VariableList list;

    list.emplace(MAX_DEPTH_NAME, OptionVariable(static_cast<int64_t>(options.maximumDepth)));
    list.emplace(SAMPLE_NAME, OptionVariable(static_cast<int64_t>(options.sampleCount)));
    list.emplace(RR_START_NAME, OptionVariable(static_cast<int64_t>(options.rrStart)));
    list.emplace(LIGHT_SAMPLER_TYPE_NAME, OptionVariable(LightSamplerCommon::LightSamplerTypeToString(options.lightSamplerType)));
    list.emplace(NEE_NAME, OptionVariable(options.nextEventEstimation));
    list.emplace(DIRECT_LIGHT_MIS_NAME, OptionVariable(options.directLightMIS));
    list.emplace(OCTREE_LEVEL_NAME, OptionVariable(static_cast<int64_t>(options.octreeLevel)));
    list.emplace(RAY_BIN_MIN_LEVEL_NAME, OptionVariable(static_cast<int64_t>(options.minRayBinLevel)));
    list.emplace(BIN_RAY_COUNT_NAME, OptionVariable(static_cast<int64_t>(options.binRayCount)));
    list.emplace(RENDER_MODE_NAME, OptionVariable(WFPGRenderModeToString(options.renderMode)));

    list.emplace(SVO_DEBUG_ITER_NAME, OptionVariable(static_cast<int64_t>(options.svoRadRenderIter)));
    list.emplace(RENDER_LEVEL_NAME, OptionVariable(static_cast<int64_t>(options.svoRenderLevel)));
    list.emplace(SVO_INIT_PATH_NAME, OptionVariable(options.svoInitPath));
    list.emplace(R_FIELD_GAUSS_ALPHA_NAME, OptionVariable(options.rFieldGaussAlpha));
    list.emplace(SKIP_PG_NAME, OptionVariable(options.skipPG));
    list.emplace(PURE_PG_NAME, OptionVariable(options.purePG));
    list.emplace(MIS_RATIO_NAME, OptionVariable(options.misRatio));
    list.emplace(PRODUCT_PG_NAME, OptionVariable(options.productPG));
    list.emplace(OPTIX_TRACE_NAME, OptionVariable(options.optiXTrace));
    list.emplace(OPTIX_USE_SCENE_NAME, OptionVariable(options.optiXUseSceneAcc));
    list.emplace(OPTIX_BUFFER_NAME, OptionVariable(static_cast<int64_t>(options.optiXBufferSize)));

    list.emplace(WEIGHTED_COMBINE_NAME, OptionVariable(options.doWeightedCombine));

    list.emplace(PARALLAX_NAME, OptionVariable(options.adjustParallax));
    list.emplace(GUIDE_BOUNCE_COUNT_NAME, OptionVariable(static_cast<int64_t>(options.guideBounceCount)));

    list.emplace(OMIT_LIGHT_NAME, OptionVariable(options.omitLightRadiance));

    list.emplace(PG_DUMP_DEBUG_NAME, OptionVariable(options.pgDumpDebugData));
    list.emplace(PG_DUMP_INTERVAL_NAME, OptionVariable(static_cast<int64_t>(options.pgDumpInterval)));
    list.emplace(PG_DUMP_PATH_NAME, OptionVariable(options.pgDumpDebugName));

    if(callbacks) callbacks->SendCurrentOptions(::Options(std::move(list)));
}

void WFPGTracer::GenerateWork(uint32_t cameraIndex)
{
    if(callbacks)
        callbacks->SendCurrentTransform(SceneCamTransform(cameraIndex));

    bool enableAA = (options.renderMode == WFPGRenderMode::RENDER ||
                     options.renderMode == WFPGRenderMode::SVO_RADIANCE);
    GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU, Vector4f>
    (
        cameraIndex,
        options.sampleCount,
        RayAuxInitWFPG(InitialWFPGAux,
                       options.sampleCount *
                       options.sampleCount),
        true,
        enableAA
    );
    // Generate Cone Aperture If
    // SVO RADIANCE mode and FALSE COLOR Mode
    if(options.renderMode == WFPGRenderMode::SVO_RADIANCE ||
       options.renderMode == WFPGRenderMode::SVO_FALSE_COLOR ||
       options.renderMode == WFPGRenderMode::SVO_NORMAL)
    {
        // Save the camera for all SVO_XXXX modes
        currentCamera.type = CameraType::SCENE_CAMERA;
        currentCamera.nonTransformedCamIndex = cameraIndex;
    }

    // On voxel trace mode we don't need paths
    if(options.renderMode == WFPGRenderMode::RENDER ||
       options.renderMode == WFPGRenderMode::SVO_RADIANCE)
        ResizeAndInitPathMemory();

    currentDepth = 0;
}

void WFPGTracer::GenerateWork(const VisorTransform& t, uint32_t cameraIndex)
{
    bool enableAA = (options.renderMode == WFPGRenderMode::RENDER ||
                     options.renderMode == WFPGRenderMode::SVO_RADIANCE);
    GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU, Vector4f>
    (
        t, cameraIndex, options.sampleCount,
        RayAuxInitWFPG(InitialWFPGAux,
                       options.sampleCount *
                       options.sampleCount),
        true,
        enableAA
    );

    // Generate Cone Aperture If
    // SVO RADIANCE mode and FALSE COLOR Mode
    if(options.renderMode == WFPGRenderMode::SVO_RADIANCE ||
       options.renderMode == WFPGRenderMode::SVO_FALSE_COLOR ||
       options.renderMode == WFPGRenderMode::SVO_NORMAL)
    {
        // Save the camera for all SVO_XXXX modes
        currentCamera.type = CameraType::TRANSFORMED_SCENE_CAMERA;
        currentCamera.transformedSceneCam.cameraIndex = cameraIndex;
        currentCamera.transformedSceneCam.transform = t;
    }

    // On voxel trace mode we don't need paths
    if(options.renderMode == WFPGRenderMode::RENDER ||
       options.renderMode == WFPGRenderMode::SVO_RADIANCE)
        ResizeAndInitPathMemory();
    currentDepth = 0;
}

void WFPGTracer::GenerateWork(const GPUCameraI& dCam)
{
    bool enableAA = (options.renderMode == WFPGRenderMode::RENDER ||
                     options.renderMode == WFPGRenderMode::SVO_RADIANCE);
    GenerateRays<RayAuxWFPG, RayAuxInitWFPG, RNGIndependentGPU, Vector4f>
    (
        dCam, options.sampleCount,
        RayAuxInitWFPG(InitialWFPGAux,
                       options.sampleCount *
                       options.sampleCount),
        true,
        enableAA
    );

    // Generate Cone Aperture If
    // SVO RADIANCE mode and FALSE COLOR Mode
    if(options.renderMode == WFPGRenderMode::SVO_RADIANCE ||
       options.renderMode == WFPGRenderMode::SVO_FALSE_COLOR ||
       options.renderMode == WFPGRenderMode::SVO_NORMAL)
    {
        // Save the camera for all SVO_XXXX modes
        currentCamera.type = CameraType::CUSTOM_CAMERA;
        currentCamera.dCustomCamera = &dCam;
    }

    // On voxel trace mode we don't need paths
    if(options.renderMode == WFPGRenderMode::RENDER ||
       options.renderMode == WFPGRenderMode::SVO_RADIANCE)
        ResizeAndInitPathMemory();
    currentDepth = 0;
}

bool WFPGTracer::Render()
{
    // Check tracer termination conditions
    // Either there is no ray left for iteration or maximum depth is exceeded
    if(rayCaster->CurrentRayCount() == 0 ||
       currentDepth >= options.maximumDepth)
        return false;

    // Don't do path tracing if debug render iteration is set
    if(options.renderMode == WFPGRenderMode::SVO_RADIANCE &&
       iterationCount >= options.svoRadRenderIter)
        return false;

    // Generate Global Data Struct
    WFPGTracerGlobalState globalData;
    globalData.gSamples = sampleMemory.GMem<Vector4f>();
    globalData.gLightList = dLights;
    globalData.totalLightCount = lightCount;
    globalData.gLightSampler = dLightSampler;
    globalData.mediumList = dMediums;
    globalData.totalMediumCount = mediumCount;
    //
    globalData.svo = svo.TreeGPU();
    globalData.gPathNodes = dPathNodes;
    globalData.maximumPathNodePerRay = MaximumPathNodePerPath();
    globalData.skipPG = options.skipPG || (currentDepth >= options.guideBounceCount);
    //
    globalData.directLightMIS = options.directLightMIS;
    globalData.nee = options.nextEventEstimation;
    globalData.rrStart = options.rrStart;

    // On voxel trace mode we just trace the rays without any material
    if(options.renderMode == WFPGRenderMode::SVO_FALSE_COLOR ||
       options.renderMode == WFPGRenderMode::SVO_NORMAL)
    {
        // Just call the voxel trace kernel on a GPU and call it a day
        LaunchDebugConeTraceKernel();
        // Signal as if we finished processing
        return false;
    }

    // Hit Rays
    rayCaster->HitRays();

    // Before Material Evaluation
    // Generate guideDirection and PDF

    if((options.renderMode != WFPGRenderMode::SVO_INITIAL_HIT_QUERY) &&
       !options.skipPG && !(currentDepth >= options.guideBounceCount))
    {
        GenerateGuidedDirections();
    }

    // Generate output partitions wrt. materials
    const auto partitions = rayCaster->PartitionRaysWRTWork();

    uint32_t totalOutRayCount = 0;
    auto outPartitions = RayCasterI::PartitionOutputRays(totalOutRayCount,
                                                         partitions,
                                                         workMap);
    // Allocate new auxiliary buffer
    // to fit all potential ray outputs
    size_t auxOutSize = totalOutRayCount * sizeof(RayAuxWFPG);
    GPUMemFuncs::EnlargeBuffer(*dAuxOut, auxOutSize);

    // Set Auxiliary Pointers
    for(auto p : outPartitions)
    {
        // Skip if null batch or not found material
        if(p.portionId == HitKey::NullBatch) continue;
        auto loc = workMap.find(p.portionId);
        if(loc == workMap.end()) continue;

        // Set pointers
        const RayAuxWFPG* dAuxInLocal = static_cast<const RayAuxWFPG*>(*dAuxIn);
        using WorkData = GPUWorkBatchD<WFPGTracerGlobalState, RayAuxWFPG>;
        int i = 0;
        for(auto& work : loc->second)
        {
            RayAuxWFPG* dAuxOutLocal = static_cast<RayAuxWFPG*>(*dAuxOut) + p.offsets[i];

            auto& wData = static_cast<WorkData&>(*work);
            wData.SetGlobalData(globalData);
            wData.SetRayDataPtrs(dAuxOutLocal, dAuxInLocal);
            i++;
        }
    }

    // Launch Kernels
    rayCaster->WorkRays(workMap, outPartitions,
                        partitions,
                        *rngCPU.get(),
                        totalOutRayCount,
                        scene.BaseBoundaryMaterial());

    // Swap auxiliary buffers since output rays are now input rays
    // for the next iteration
    SwapAuxBuffers();
    // Increase Depth
    currentDepth++;
    return true;
}

void WFPGTracer::Finalize()
{
    // Iteration count is used to when dump the entire svo
    // to the disk etc.
    iterationCount++;

    //Debug::DumpBatchedMemToFile("PathNodes", dPathNodes,
    //                            MaximumPathNodePerPath(),
    //                            TotalPathNodeCount());

    // Deposit the radiances on the path chains
    if(options.renderMode == WFPGRenderMode::RENDER ||
       options.renderMode == WFPGRenderMode::SVO_RADIANCE)
    {

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        uint32_t totalPathNodeCount = TotalPathNodeCount();
        svo.AccumulateRaidances(dPathNodes, totalPathNodeCount,
                                MaximumPathNodePerPath(), cudaSystem);
        svo.NormalizeAndFilterRadiance(cudaSystem);

        float milliseconds = 0;
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        METU_LOG("Cache; -; {:f}", milliseconds);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // Dump the SVO tree if requested
        uint32_t dumpInterval = static_cast<uint32_t>(std::pow(options.pgDumpInterval, treeDumpCount));
        if(options.pgDumpDebugData && iterationCount == dumpInterval)
        {
            std::vector<Byte> svoData;
            svo.DumpSVOAsBinary(svoData, cudaSystem);
            std::string fName = fmt::format("{:d}_{:s}",
                                            iterationCount,
                                            options.pgDumpDebugName);
            Utility::DumpStdVectorToFile(svoData, fName);
            METU_LOG("Dumping {:s}", fName);
            treeDumpCount++;
        }

    }
    // On SVO_Radiance mode clear the image memory
    // And trace the SVO from the camera and send the results
    // On voxel trace mode we just trace the rays without any material
    if(options.renderMode == WFPGRenderMode::SVO_RADIANCE)
    {
        // Clear the image buffer
        imgMemory.Reset(cudaSystem);
        LaunchDebugConeTraceKernel();

        // Completely Reset the Image
        // This is done to eliminate old data from prev samples
        if(callbacks)
        {
            Vector2i start = imgMemory.SegmentOffset();
            Vector2i end = start + imgMemory.SegmentSize();
            callbacks->SendImageSectionReset(start, end);
        }
    }

    METU_LOG("----------------");
    cudaSystem.SyncAllGPUs();
    frameTimer.Split();

    double ms = frameTimer.Elapsed<CPUTimeMillis>();
    METU_LOG("TOTAL:{}", ms);


    //if(iterationCount == 2)
    //ResetImage();

    UpdateFrameAnalytics("paths / sec", options.sampleCount * options.sampleCount);


    if(options.doWeightedCombine)
    {
        if(crashed) return;
        const auto sampleGMem = std::as_const(sampleMemory).GMem<Vector4f>();

        //
        float scalarSampleMultiplier = (iterationCount == 1) ? 1.0f : 2.0f;


        //    uint32_t multiplier = std::min(4u, iterationCount - 1);
        ////uint32_t scalarWeightInt = (1 << multiplier);
        //uint32_t scalarWeightInt = multiplier + 1;

        //float scalarSampleMultiplier = static_cast<float>(scalarWeightInt);
        METU_LOG("Scalar Weight: {}", scalarSampleMultiplier);

        if(reconFilter != nullptr)
            reconFilter->FilterToImg(imgMemory,
                                     sampleGMem.gValues,
                                     sampleGMem.gImgCoords,
                                     sampleMemory.SampleCount(),
                                     cudaSystem,
                                     scalarSampleMultiplier);
        else
        {
            METU_ERROR_LOG("Reconstruction Filter did not set!");
            crashed = true;
        }
        GPUTracer::Finalize();
    }
    else
    {
        RayTracer::Finalize();
    }

    using namespace std::chrono_literals;
    //std::this_thread::sleep_for(10s);
}

size_t WFPGTracer::TotalGPUMemoryUsed() const
{
    return (RayTracer::TotalGPUMemoryUsed() +
            svo.UsedGPUMemory() +
            lightSamplerMemory.Size() + pathMemory.Size());
}