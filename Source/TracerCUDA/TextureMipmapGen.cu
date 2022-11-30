#include "TextureMipmapGen.cuh"

#include "CudaSystem.h"
#include "CudaSystem.hpp"
#include "GPUReconFilterMitchell.h"
#include "TextureMipmapGen.cuh"
#include "TextureReference.cuh"

#include <cub/cub.cuh>

template <uint32_t TPB, class WriteType, class FilterFunc>
__global__
static void KCGenMipmap(// Outputs
                        cudaSurfaceObject_t sObj,
                        // Inputs
                        TextureRef<2, TexFloatType_t<2>> texture,
                        // Constants
                        Vector2ui mipTexSize,
                        Vector2ui texSize,
                        uint32_t mipLevel,
                        uint32_t sampleXY,
                        float filterRadius,
                        FilterFunc filter)
{
    // TODO: Put this on a library
    // https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_16.pdf
    auto CocentricDiskSample = [=](const Vector2f& uv) -> Vector2f
    {
        const float R = filterRadius;
        float a = 2 * uv[0] - 1;
        float b = 2 * uv[1] - 1;
        if(b == 0) b = 1;

        float r;
        float phi;
        if(a * a > b * b)
        {
            r = R * a;
            phi = (MathConstants::Pi * 0.25f) * (b / a);
        }
        else
        {
            r = R * b;
            phi = ((MathConstants::Pi * 0.5) -
                   (MathConstants::Pi * 0.25f) * (a / b));
        }
        // Convert to relative pixel index
        return Vector2f(r * cos(phi),
                        r * sin(phi));
    };

    using ReadType = TexFloatType_t<2>;
    static constexpr uint32_t WARP_PER_BLOCK = TPB / WARP_SIZE;
    using WarpValueReduce = cub::WarpReduce<ReadType>;
    using WarpFloatReduce = cub::WarpReduce<float>;

    union SharedMemory
    {
        typename WarpValueReduce::TempStorage valReduceMem[WARP_PER_BLOCK];
        typename WarpValueReduce::TempStorage weightReduceMem[WARP_PER_BLOCK];
    };

    __shared__ SharedMemory shMem;

    // Each warp is responsible for single pixel on the leaf
    const uint32_t kernelWarpCount = (blockDim.x * gridDim.x) / WARP_SIZE;
    const uint32_t pixelCount = mipTexSize.Multiply();

    uint32_t globalThreadId = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t globalWarpId = globalThreadId / WARP_SIZE;
    uint32_t localWarpId = threadIdx.x / WARP_SIZE;
    uint32_t laneId = globalThreadId % WARP_SIZE;

    uint32_t samplePerPixel = sampleXY * sampleXY;
    uint32_t warpLoopCount = (samplePerPixel + WARP_SIZE - 1) / WARP_SIZE;

    // Warp stride Loop
    for(uint32_t pixelId = globalWarpId; pixelId < pixelCount;
        pixelId += kernelWarpCount)
    {
        Vector2ui pixIndex2D = Vector2ui(pixelId % mipTexSize[0],
                                         pixelId / mipTexSize[0]);
        Vector2f pixCenterFloat = Vector2f(pixIndex2D) + 0.5f;
        Vector2f mipZeroUV = pixCenterFloat / Vector2f(mipTexSize);


        ReadType leaderTotal = ReadType{0};
        float leaderWeightTotal = 0.0f;

        // Do not loop over using laneId and warpSize
        // reduction operation may block because of inactive threads
        for(uint32_t i = 0; i < warpLoopCount; i++)
        {
            // Generate fake random variables
            // stratify 2D [0,1) space wrt. total sample count
            uint32_t localSampleId = i * WARP_SIZE + laneId;
            bool isValid = (localSampleId < samplePerPixel);
            Vector2ui laneStratId(localSampleId % sampleXY,
                                  localSampleId / sampleXY);

            Vector2f sampleXi = (Vector2f(laneStratId) + Vector2f(0.5f)) / Vector2f(sampleXY);

            Vector2f sampleOffset = CocentricDiskSample(sampleXi);

            // Current filter functions requires filter center coordinates
            // and the filtering location image space coordinate
            float weight = filter(pixCenterFloat + sampleOffset,
                                  pixCenterFloat);

            // UV of the miplevel0 is required scale offset accordingly
            sampleOffset *= static_cast<float>(1 << mipLevel);
            Vector2f texReadUV = mipZeroUV + sampleOffset / Vector2f(texSize);
            ReadType texVal = texture(texReadUV, static_cast<float>(mipLevel - 1));
            ReadType weightedVal = texVal * weight;

            texVal = (isValid) ? texVal : ReadType{0};
            weight = (isValid) ? weight : 0.0f;

            leaderTotal += WarpValueReduce(shMem.valReduceMem[localWarpId]).Sum(texVal);
            leaderWeightTotal += WarpWeightReduce(shMem.weightReduceMem).Sum(weight);
        }

        // Finally leader writes to the surface object
        if(laneId == 0)
        {
            ReadType total = leaderTotal / leaderWeightTotal;

            // TODO: do conversion here
            //surf2DWrite(total, sObj, pixIndex2D[0] * sizeof(WriteType), pixIndex2D[1]);
        }
    }

}

__host__
Texture<2, Vector4f> GenerateMipmaps(const Texture<2, Vector4f>& texture, uint32_t upToMip)
{
    const CudaGPU& textureGPU = *texture.Device();
    std::vector<CudaSurfaceRAII> surfaces;
    CUDA_CHECK(cudaSetDevice(texture.Device()->DeviceId()));

    // Allocate new mipmapped texture
    Texture<2, Vector4f> newTexture = texture.EmptyMipmappedTexture(upToMip);

    // Sample stratified MULTISAMPLE_COUNT * MULTISAMPLE_COUNT
    // amount of samples over the region of the texel.
    // Filter these according to the filter
    static constexpr float FILTER_RADIUS = 2.0f;
    static constexpr uint32_t MULTISAMPLE_COUNT = 5;
    // Mitchell-Netravali Filter
    const GPUMitchellFilterFunctor filterFunctor(FILTER_RADIUS, 0.3333f, 0.3333f);

    // Construct mips level by level
    upToMip = std::min(upToMip, newTexture.MipmapCount() - 1);
    for(uint32_t mipLevel = 1; mipLevel <= upToMip; mipLevel++)
    {
        CudaSurfaceRAII surfaceObject = newTexture.GetMipLevelSurface(mipLevel);
        // Find out the return type
        Vector2ui mipDim = Vector2ui::Max(texture.Dimensions() / (1 << mipLevel), Vector2ui(1));

        METU_LOG("{}, {}", mipDim[0], mipDim[1]);

        static constexpr uint32_t TPB = StaticThreadPerBlock1D;
        static constexpr uint32_t WARP_PER_BLOCK = TPB / WARP_SIZE;
        uint32_t totalThreadCount = mipDim.Multiply() * MULTISAMPLE_COUNT * MULTISAMPLE_COUNT;
        uint32_t totalWarpCount = (totalThreadCount + WARP_SIZE - 1) / WARP_SIZE;
        // TODO: make this utilize less blocks for multi kernel execution
        // Currently, it generates full amount of blocks
        uint32_t totalBlockCount = totalWarpCount / WARP_PER_BLOCK;

        textureGPU.ExactKC_X(0, (cudaStream_t)0,
                             TPB, totalBlockCount,
                             //
                             KCGenMipmap<StaticThreadPerBlock1D, Vector4f, GPUGaussianFilterFunctor>,
                             // Output
                             surface,
                             // Inputs
                             TextureRef<2, Vector3f>(newTexture),
                             // Constants
                             mipDim,
                             texture.Dimensions(),
                             static_cast<uint32_t>(mipLevel),
                             MULTISAMPLE_COUNT,
                             FILTER_RADIUS,
                             filterFunctor);


        // Defer destruction of the surface object until all kernels are finished
        surfaces.emplace_back(std::move(surfaceObject));
    }

    // Wait all events to finish before deleting surfaces (implicit)
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)0));

    return newTexture;

}