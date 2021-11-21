#include "GPUTransformSingle.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

#include "RayLib/SceneNodeNames.h"
#include "RayLib/MemoryAlignment.h"

inline __device__ void ConstrucctTransform(GPUTransformSingle* gTransformLocation,
                                           const Matrix4x4& transform,
                                           const Matrix4x4& invTransform)
{
    new (gTransformLocation) GPUTransformSingle(transform, invTransform);
}

__global__ void KCConstructGPUTransform(GPUTransformSingle* gTransformLocations,
                                        Matrix4x4* gInvTransforms,
										const Matrix4x4* gTransforms,
										uint32_t transformCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < transformCount;
        globalId += blockDim.x * gridDim.x)
    {
        const Matrix4x4& transform = gTransforms[globalId];
        Matrix4x4& invTransform = gInvTransforms[globalId];

        // Invert the transform
        invTransform = transform.Inverse();

        // Allocate class using transform and inverted transform
        ConstrucctTransform(gTransformLocations + globalId,
                            transform, invTransform);
    }
}

SceneError CPUTransformSingle::InitializeGroup(const NodeListing& transformNodes,
											   double time,
											   const std::string&)
{
    std::vector<Matrix4x4> transforms;
    for(const auto& node : transformNodes)
    {
        std::string layoutName = node->CommonString(LAYOUT);

        std::vector<Matrix4x4> nodeTransforms;
        if(layoutName == LAYOUT_MATRIX)
        {
            nodeTransforms = node->AccessMatrix4x4(MATRIX, time);
        }
        else if(layoutName == LAYOUT_TRS)
        {
            nodeTransforms.reserve(node->IdCount());
            nodeTransforms.clear();

            std::vector<Vector3> translations = node->AccessVector3(TRANSLATE, time);
            std::vector<Vector3> rotations = node->AccessVector3(ROTATE, time);
            std::vector<Vector3> scales = node->AccessVector3(SCALE, time);
            assert(translations.size() == rotations.size());
            assert(rotations.size() == scales.size());

            for(uint32_t i = 0; i < translations.size(); i++)
            {
                Vector3 r = rotations[i];
                const Vector3& s = scales[i];
                const Vector3& t = translations[i];
                r *= MathConstants::DegToRadCoef;

                // Convert to Matrix
                Matrix4x4 m = TransformGen::Rotate(r[0], XAxis);
                m = TransformGen::Rotate(r[1], YAxis) * m;
                m = TransformGen::Rotate(r[2], ZAxis) * m;
                m = TransformGen::Scale(s[0], s[1], s[2]) * m;
                m = TransformGen::Translate(t) * m;
                nodeTransforms.push_back(std::move(m));
            }
        }
        else return SceneError::TRANSFORM_TYPE_INTERNAL_ERROR;

        transforms.insert(transforms.end(),
                          nodeTransforms.begin(),
                          nodeTransforms.end());
    }

    // Generated/Loaded matrices
    // Allocate transform class and matrix on GPU
    transformCount = static_cast<uint32_t>(transforms.size());
    size_t sizeOfMatrices = sizeof(Matrix4x4) * transformCount;
    sizeOfMatrices = Memory::AlignSize(sizeOfMatrices);
    size_t sizeOfTransformClasses = sizeof(GPUTransformSingle) * transformCount;
    sizeOfTransformClasses = Memory::AlignSize(sizeOfTransformClasses);

    size_t requiredSize = (sizeOfMatrices * 2 +
                           sizeOfTransformClasses);

    // Reallocate if memory is not enough
    GPUMemFuncs::EnlargeBuffer(memory, requiredSize);

    size_t offset = 0;
    std::uint8_t* dBasePtr = static_cast<uint8_t*>(memory);
    dTransformMatrices = reinterpret_cast<Matrix4x4*>(dBasePtr + offset);
    offset += sizeOfMatrices;
    dInvTransformMatrices = reinterpret_cast<Matrix4x4*>(dBasePtr + offset);
    offset += sizeOfMatrices;
    dGPUTransforms = reinterpret_cast<GPUTransformSingle*>(dBasePtr + offset);
    offset += sizeOfTransformClasses;
    assert(requiredSize == offset);

    // Copy
    CUDA_CHECK(cudaMemcpy(const_cast<Matrix4x4*>(dTransformMatrices),
                          transforms.data(),
                          transformCount * sizeof(Matrix4x4),
                          cudaMemcpyHostToDevice));
    return SceneError::OK;
}

SceneError CPUTransformSingle::ChangeTime(const NodeListing&, double,
										  const std::string&)
{
    // Change time system have not implemented yet
    return SceneError::UNKNOWN_TRANSFORM_TYPE;
}

TracerError CPUTransformSingle::ConstructTransforms(const CudaSystem& system)
{
    // Call allocation kernel
    const CudaGPU& gpu = system.BestGPU();
    CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
    gpu.GridStrideKC_X(0, 0,
                       TransformCount(),
                       //
                       KCConstructGPUTransform,
                       //
                       const_cast<GPUTransformSingle*>(dGPUTransforms),
                       const_cast<Matrix4x4*>(dInvTransformMatrices),
                       dTransformMatrices,
                       TransformCount());

    gpu.WaitMainStream();

    // Generate transform list
    for(uint32_t i = 0; i < TransformCount(); i++)
    {
        const auto* ptr =  static_cast<const GPUTransformI*>(dGPUTransforms + i);
        gpuTransformList.push_back(ptr);
    }
    return TracerError::OK;
}