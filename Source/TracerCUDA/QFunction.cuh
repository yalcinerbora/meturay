#pragma once

#include "RayLib/Vector.h"
#include "DeviceMemory.h"
#include "RNGenerator.h"

#include "DataStructsCommon.cuh"
#include "GPUPiecewiseDistribution.cuh"

class CudaSystem;
class QFunctionCPU;

class QFunctionGPU
{
    friend class QFunctionCPU;

    private:
        PWCDistStaticGPU2D  gDistributions;
        float*              gQFunction;
        Vector2ui           dataPerNode;
        uint32_t            nodeCount;
        float               alpha;

        __device__
        float*              AcquireData(const Vector2ui& dirIndex,
                                        uint32_t spatialIndex);
        __device__
        const float*        AcquireData(const Vector2ui& dirIndex,
                                        uint32_t spatialIndex) const;

    public:
        // Constructors & Destructor
                        QFunctionGPU();
                        QFunctionGPU(PWCDistStaticGPU2D gDistributions,
                                     float* gQFunction,
                                     const Vector2ui& dataPerNode,
                                     uint32_t nodeCount,
                                     float alpha);
                        ~QFunctionGPU() = default;

        // Interface
        __device__
        Vector2ui       DirectionalRes() const;

        __device__
        Vector3f        Sample(float& pdf, RNGeneratorGPUI& rng,
                               uint32_t spatialIndex) const;
        __device__
        float           Pdf(const Vector3f& worldDir, uint32_t spatialIndex) const;

        __device__
        Vector3f        Direction(const Vector2ui& dirIndex) const;

        // Access Value of the certain strata
        __device__
        float           Value(const Vector2ui& dirIndex,
                              uint32_t spatialIndex) const;
        // Access Value of the closest strata given a direction
        __device__
        float           Value(const Vector3f& worldDir,
                              uint32_t spatialIndex) const;

        // Atomically update a certain strata with the given value
        __device__
        Vector3f        Update(const Vector3f& worldDir,
                               float radiance,
                               uint32_t spatialIndex);
};

class QFunctionCPU
{
    private:
        DeviceMemory            memory;
        QFunctionGPU            qFuncGPU;
        PWCDistStaticCPU2D      distributions;
        uint32_t                spatialCount;

        static constexpr float  INITIAL_Q_VALUE = 0.0f;

    protected:
    public:
        // Constructors & Destructor
                        QFunctionCPU() = default;
                        QFunctionCPU(float alpha,
                                     const Vector2ui& dataPerNode,
                                     uint32_t spatialCount);
                        QFunctionCPU(const QFunctionCPU&) = delete;
        QFunctionCPU&   operator=(const QFunctionCPU&) = delete;
        QFunctionCPU&   operator=(QFunctionCPU&&) = default;
                        ~QFunctionCPU() = default;

        // Methods
        TracerError     Initialize(const CudaSystem&);
        void            RecalculateDistributions(const CudaSystem&);
        QFunctionGPU    FunctionGPU() const;

        void            DumpFunctionAsBinary(std::vector<Byte>&) const;

        size_t          UsedGPUMemory() const;
        size_t          UsedCPUMemory() const;
};

__device__ inline
float* QFunctionGPU::AcquireData(const Vector2ui& dirIndex,
                                 uint32_t spatialIndex)
{
    return gQFunction + (spatialIndex * dataPerNode.Multiply() +
                         dirIndex[1] * dataPerNode[0] +
                         dirIndex[0]);
}

__device__ inline
const float* QFunctionGPU::AcquireData(const Vector2ui& dirIndex,
                                       uint32_t spatialIndex) const
{
    return gQFunction + (spatialIndex * dataPerNode.Multiply() +
                         dirIndex[1] * dataPerNode[0] +
                         dirIndex[0]);
}

inline
QFunctionGPU::QFunctionGPU()
    : gQFunction(nullptr)
    , dataPerNode(Zero2ui)
    , nodeCount(0)
    , alpha(0.0f)
{}

inline
QFunctionGPU::QFunctionGPU(PWCDistStaticGPU2D gDistributions,
                           float* gQFunction,
                           const Vector2ui& dataPerNode,
                           uint32_t nodeCount,
                           float alpha)
    : gDistributions(gDistributions)
    , gQFunction(gQFunction)
    , dataPerNode(dataPerNode)
    , nodeCount(nodeCount)
    , alpha(alpha)
{}

__device__ inline
Vector2ui QFunctionGPU::DirectionalRes() const
{
    return dataPerNode;
}

__device__ inline
Vector3f QFunctionGPU::Sample(float& pdf, RNGeneratorGPUI& rng,
                              uint32_t spatialIndex) const
{
    Vector2f index;
    gDistributions.Sample(pdf, index, rng, spatialIndex);
    index /= Vector2f(dataPerNode[0], dataPerNode[1]);

    return GPUDataStructCommon::DiscreteCoordsToDir(pdf, index);
}

__device__ inline
float QFunctionGPU::Pdf(const Vector3f& worldDir,
                        uint32_t spatialIndex) const
{
    float pdfDir = 1.0f;
    Vector2f coords = GPUDataStructCommon::DirToDiscreteCoords(pdfDir, worldDir);
    coords *= Vector2f(dataPerNode[0], dataPerNode[1]);
    //if(coords[0] >= dataPerNode[0] ||
    //   coords[1] >= dataPerNode[1])
    //{
    //    printf("OOR [%f, %f]\n", coords[0], coords[1]);
    //    return 1.0f;
    //}
    //if(isnan(pdfDir) || isinf(pdfDir))
    //{
    //    printf("[%f] NaN or Inf pdfDir on dirToDiscrete (%f, %f, %f)\n",
    //           pdfDir, worldDir[0], worldDir[1], worldDir[2]);
    //    return 0;
    //}

    float pdfDist = gDistributions.Pdf(coords, spatialIndex);
    //if(isnan(pdfDist) || isinf(pdfDist))
    //{
    //    printf("[%f] NaN or Inf pdfDist on [%f, %f] qIndex [%d]\n",
    //           pdfDist, coords[0], coords[1], spatialIndex);
    //    return 0;
    //}
    return pdfDir * pdfDist;
}

__device__ inline
Vector3f QFunctionGPU::Direction(const Vector2ui& dirIndex) const
{
    Vector2f indexF = Vector2f(dirIndex[0], dirIndex[1]);
    // Get the center of the "pixel"
    indexF += Vector2f(0.5f);
    // Normalize
    indexF /= Vector2f(dataPerNode[0], dataPerNode[1]);
    // Convert to spherical coordinates
    return GPUDataStructCommon::DiscreteCoordsToDir(indexF);
}

__device__ inline
float QFunctionGPU::Value(const Vector2ui& dirIndex,
                          uint32_t spatialIndex) const
{
    const float* location = AcquireData(dirIndex, spatialIndex);
    // Here we can read the value directly
    // Data may not be recent (since these are updated by other threads
    // atomically), however it should still converge
    return *location;
}

__device__ inline
float QFunctionGPU::Value(const Vector3f& worldDir,
                          uint32_t spatialIndex) const
{
    Vector2f uv = GPUDataStructCommon::DirToDiscreteCoords(worldDir);
    Vector2f indexF = uv * Vector2f(dataPerNode[0], dataPerNode[1]);
    Vector2ui index = Vector2ui(static_cast<int32_t>(indexF[0]),
                                static_cast<int32_t>(indexF[1]));
    const float* location = AcquireData(index, spatialIndex);
    // Here we can read the value directly
    // Data may not be recent (since these are updated by other threads
    // atomically), however it should still converge
    return *location;
}

__device__ inline
Vector3f QFunctionGPU::Update(const Vector3f& worldDir,
                              float radiance,
                              uint32_t spatialIndex)
{
    // Convert to UV Coordinates
    Vector2f uv = GPUDataStructCommon::DirToDiscreteCoords(worldDir);
    Vector2f indexF = uv * Vector2f(dataPerNode[0], dataPerNode[1]);
    Vector2ui index = Vector2ui(static_cast<int32_t>(indexF[0]),
                              static_cast<int32_t>(indexF[1]));
    float* location = AcquireData(index, spatialIndex);
    // Atomic Q-Update using CAS Atomics
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
    static_assert(sizeof(float) == sizeof(uint32_t), "Sanity Check");
    uint32_t* locUInt = reinterpret_cast<uint32_t*>(location);
    uint32_t assumed;
    uint32_t old = *locUInt;
    do
    {
        assumed = old;
        float assumedFloat = __uint_as_float(assumed);
        // Actual Operation
        float result = ((1.0f - alpha) * assumedFloat +
                        alpha * radiance);
        uint32_t newVal = __float_as_uint(result);
        old = atomicCAS(locUInt, assumed, newVal);
    }
    while(assumed != old);
}

inline
QFunctionCPU::QFunctionCPU(float alpha,
                           const Vector2ui& dataPerNode,
                           uint32_t spatialCount)
    : qFuncGPU()
    , spatialCount(spatialCount)
{
    // Allocate the Data
    GPUMemFuncs::AllocateMultiData(std::tie(qFuncGPU.gQFunction),
                                   memory,
                                   {dataPerNode.Multiply() * spatialCount});

    qFuncGPU.dataPerNode = dataPerNode;
    qFuncGPU.nodeCount = spatialCount;
    qFuncGPU.alpha = alpha;
}

inline
QFunctionGPU QFunctionCPU::FunctionGPU() const
{
    return qFuncGPU;
}

inline
size_t QFunctionCPU::UsedGPUMemory() const
{
    return memory.Size() + distributions.UsedGPUMemory();
}

inline
size_t QFunctionCPU::UsedCPUMemory() const
{
    return sizeof(QFunctionCPU);
}