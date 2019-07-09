#include "GPUPrimitiveSphere.h"

#include "RayLib/PrimitiveDataTypes.h"
#include "RayLib/SurfaceDataIO.h"
#include "RayLib/SceneError.h"
#include "RayLib/SceneNodeI.h"

// Generics
GPUPrimitiveSphere::GPUPrimitiveSphere()
    : totalPrimitiveCount(0)
{}

const char* GPUPrimitiveSphere::Type() const
{
    return TypeName();
}

SceneError GPUPrimitiveSphere::InitializeGroup(const NodeListing& surfaceDataNodes,
                                               double time)
{
    std::vector<std::vector<size_t>> primCountList;

    // Generate Loaders
    std::vector<std::unique_ptr<SurfaceDataLoaderI>> loaders;
    for(const auto& sPtr : surfaceDataNodes)
    {
        const SceneNodeI& s = *sPtr;
        loaders.push_back(std::move(SurfaceDataIO::GenSurfaceDataLoader(s, time)));
    }

    SceneError e = SceneError::OK;
    totalPrimitiveCount = 0;
    for(const auto& loader : loaders)
    {
        const SceneNodeI& node = loader->SceneNode();
        const size_t batchCount = node.IdCount();

        std::vector<AABB3>  aabbList(batchCount);
        primCountList.emplace_back(batchCount);

        // Load Data
        if((e = loader->PrimitiveCount(primCountList.back().data())) != SceneError::OK)
           return e;
        if((e = loader->AABB(aabbList.data())) != SceneError::OK)
            return e;

        size_t i = 0;
        for(const auto& pair : node.Ids())
        {
            NodeId id = pair.first;

            uint64_t start = totalPrimitiveCount;
            uint64_t end = start + primCountList.back()[i];
            totalPrimitiveCount = end;

            batchRanges.emplace(id, Vector2ul(start, end));
            batchAABBs.emplace(id, aabbList[i]);

            i++;
        }
    }

    std::vector<float> postitionsCPU(totalPrimitiveCount * 3);
    std::vector<float> radiusCPU(totalPrimitiveCount);
    size_t offset = 0;
    size_t i = 0;
    for(const auto& loader : loaders)
    {
        const SceneNodeI& node = loader->SceneNode();
        const size_t batchCount = node.IdCount();

        size_t j = 0;
        for(const auto& pair : node.Ids())
        {
            if(e != loader->GetPrimitiveData(reinterpret_cast<Byte*>(postitionsCPU.data() + offset),
                                             PrimitiveDataType::POSITION))
                return e;
            if(e != loader->GetPrimitiveData(reinterpret_cast<Byte*>(radiusCPU.data() + offset),
                                             PrimitiveDataType::RADIUS))
                return e;

            offset += primCountList[i][j];
            j++;
        }
    }
    assert(offset == totalPrimitiveCount);

    // All loaded to CPU
    // Now copy to GPU
    // Alloc
    memory = std::move(DeviceMemory(sizeof(Vector4f) * totalPrimitiveCount));
    float* dCentersRadius = static_cast<float*>(memory);

    CUDA_CHECK(cudaMemcpy2D(dCentersRadius, sizeof(Vector4f),
                            postitionsCPU.data(), sizeof(float) * 3,
                            sizeof(float) * 3, totalPrimitiveCount,
                            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy2D(dCentersRadius + 3, sizeof(Vector4f),
                            radiusCPU.data(), sizeof(float),
                            sizeof(float), totalPrimitiveCount,
                            cudaMemcpyHostToDevice));

    // Set Main Pointers of batch
    dData.centerRadius = reinterpret_cast<Vector4f*>(dCentersRadius);
    return e;
}

SceneError GPUPrimitiveSphere::ChangeTime(const NodeListing& surfaceDataNodes, double time)
{
    std::vector<std::vector<size_t>> primCountList;

    // Generate Loaders
    std::vector<std::unique_ptr<SurfaceDataLoaderI>> loaders;
    for(const auto& sPtr : surfaceDataNodes)
    {
        const SceneNodeI& s = *sPtr;
        loaders.push_back(std::move(SurfaceDataIO::GenSurfaceDataLoader(s, time)));
    }

    // First update aabbs (these should have changed)
    SceneError e = SceneError::OK;
    totalPrimitiveCount = 0;
    for(const auto& loader : loaders)
    {
        const SceneNodeI& node = loader->SceneNode();
        const size_t batchCount = node.IdCount();

        std::vector<AABB3>  aabbList(batchCount);
        primCountList.emplace_back(batchCount);

        // Load Data
        if((e = loader->PrimitiveCount(primCountList.back().data())) != SceneError::OK)
            return e;
        if((e = loader->AABB(aabbList.data())) != SceneError::OK)
            return e;

        size_t i = 0;
        for(const auto& pair : node.Ids())
        {
            NodeId id = pair.first;
            const Vector2ul range = batchRanges.at(id);
            assert((range[1] - range[0]) == primCountList.back()[i]);
            batchAABBs.at(id) = aabbList[i];

            i++;
        }
    }

    // Now Copy

////
//    SceneError e = SceneError::OK;
//    std::vector<float> postitionsCPU, radiusCPU;
//    for(const auto& loader : loaders)
//    {
//        const SceneNodeI& node = loader->SceneNode();
//        const size_t batchCount = node.IdCount();
//
//        size_t i = 0;
//        for(const auto& pair : node.Ids())
//        {
//            NodeId id = pair.first;
//            Vector2ul range = batchRanges[id];
//
//
//            size_t primitiveCount = loader->PrimitiveCount();
//            assert((range[1] - range[0]) == primitiveCount);
//
//            postitionsCPU.resize(primitiveCount * 3);
//            radiusCPU.resize(primitiveCount);
//
//            if(e != loader->GetPrimitiveData(reinterpret_cast<Byte*>(postitionsCPU.data()),
//                                             PrimitiveDataType::POSITION))
//                return e;
//            if(e != loader->GetPrimitiveData(reinterpret_cast<Byte*>(radiusCPU.data()),
//                                             PrimitiveDataType::RADIUS))
//                return e;
//
//            // Copy
//            float* dCentersRadius = static_cast<float*>(memory);
//            CUDA_CHECK(cudaMemcpy2D(dCentersRadius, sizeof(Vector4f),
//                                    postitionsCPU.data(), sizeof(float) * 3,
//                                    sizeof(float) * 3, primitiveCount,
//                                    cudaMemcpyHostToDevice));
//
//            CUDA_CHECK(cudaMemcpy2D(dCentersRadius + 3, sizeof(Vector4f),
//                                    radiusCPU.data(), sizeof(float),
//                                    sizeof(float), primitiveCount,
//                                    cudaMemcpyHostToDevice));
//        }
//    }
    return e;
}

Vector2ul GPUPrimitiveSphere::PrimitiveBatchRange(uint32_t surfaceDataId) const
{
    return batchRanges.at(surfaceDataId);
}

AABB3 GPUPrimitiveSphere::PrimitiveBatchAABB(uint32_t surfaceDataId) const
{
    return batchAABBs.at(surfaceDataId);
}

bool GPUPrimitiveSphere::CanGenerateData(const std::string& s) const
{
    return (s == PrimitiveDataTypeToString(PrimitiveDataType::POSITION) ||
            s == PrimitiveDataTypeToString(PrimitiveDataType::NORMAL) ||
            s == PrimitiveDataTypeToString(PrimitiveDataType::UV));
}