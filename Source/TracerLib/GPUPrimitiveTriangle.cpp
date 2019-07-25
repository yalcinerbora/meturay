#include "GPUPrimitiveTriangle.h"

#include "RayLib/PrimitiveDataTypes.h"
#include "RayLib/SceneError.h"
#include "RayLib/Log.h"

#include "RayLib/SurfaceLoaderGenerator.h"
#include "RayLib/SurfaceLoaderI.h"

// Generics
GPUPrimitiveTriangle::GPUPrimitiveTriangle()
    : totalPrimitiveCount(0)
    , totalDataCount(0)
{}

const char* GPUPrimitiveTriangle::Type() const
{
    return TypeName();
}

SceneError GPUPrimitiveTriangle::InitializeGroup(const NodeListing& surfaceDataNodes, double time,
                                                 const SurfaceLoaderGeneratorI& loaderGen)
{
    SceneError e = SceneError::OK;
    std::vector<std::vector<size_t>> primCountList;
    std::vector<std::vector<size_t>> primDataCountList;    

    // Generate Loaders
    std::vector<SharedLibPtr<SurfaceLoaderI>> loaders;
    for(const auto& sPtr : surfaceDataNodes)
    {
        const SceneNodeI& s = *sPtr;
        SharedLibPtr<SurfaceLoaderI> sl(nullptr, [] (SurfaceLoaderI*)->void {});        
        if((e = loaderGen.GenerateSurfaceLoader(sl, s, time)) != SceneError::OK)
           return e;
        loaders.emplace_back(std::move(sl));
    }

    totalPrimitiveCount = 0;
    totalDataCount = 0;
    for(const auto& loader : loaders)
    {
        const SceneNodeI& node = loader->SceneNode();
        const size_t batchCount = node.IdCount();

        std::vector<AABB3> aabbList(batchCount);
        primCountList.emplace_back(batchCount);
        primDataCountList.emplace_back(batchCount);

        // Load Aux Data
        if((e = loader->PrimitiveCounts(primCountList.back().data())) != SceneError::OK)
            return e;
        if((e = loader->PrimitiveDataCount(primDataCountList.back().data(), PrimitiveDataType::POSITION)) != SceneError::OK)
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

            uint64_t dataStart = totalDataCount;
            uint64_t dataEnd = dataStart + primDataCountList.back()[i];
            totalDataCount = dataEnd;

            batchRanges.emplace(id, Vector2ul(start, end));
            batchDataRanges.emplace(id, Vector2ul(dataStart, dataEnd));
            batchAABBs.emplace(id, aabbList[i]);

            i++;
        }
    }

    // Now allocate to CPU then GPU
    const size_t totalVertexCount = totalPrimitiveCount * 3;
    const size_t totalVertexDataCount = totalDataCount;
    std::vector<float> postitionsCPU(totalVertexDataCount * 3);
    std::vector<float> normalsCPU(totalVertexDataCount * 3);
    std::vector<float> uvsCPU(totalVertexDataCount * 2);
    std::vector<uint32_t> indexCPU(totalVertexCount);
    size_t offsetData = 0;
    size_t offsetIndex = 0;
    size_t i = 0;
    for(const auto& loader : loaders)
    {
        const SceneNodeI& node = loader->SceneNode();
        const size_t batchCount = node.IdCount();

        if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(postitionsCPU.data() + (offsetData * 3)),
                                         PrimitiveDataType::POSITION)) != SceneError::OK)
            return e;
        if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(normalsCPU.data() + (offsetData * 3)),
                                         PrimitiveDataType::NORMAL)) != SceneError::OK)
            return e;
        if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(uvsCPU.data() + (offsetData * 2)),
                                         PrimitiveDataType::UV)) != SceneError::OK)
            return e;
        if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(indexCPU.data() + (offsetIndex * 3)),
                                         PrimitiveDataType::VERTEX_INDEX)) != SceneError::OK)
            return e;

        // Generate offset
        size_t j = 0;
        for(const auto& pair : node.Ids())
        {
            offsetData += primDataCountList[i][j];
            offsetIndex += primCountList[i][j];
            j++;
        }
    }
    assert(offsetIndex == totalPrimitiveCount);
    assert(offsetData == totalDataCount);

    // All loaded to CPU, copy to GPU
    // Alloc
    memory = std::move(DeviceMemory(sizeof(Vector4f) * 2 * totalVertexDataCount +
                                    sizeof(uint32_t) * totalVertexCount));
    float* dPositionsU = static_cast<float*>(memory);
    float* dNormalsV = static_cast<float*>(memory) + totalVertexDataCount * 4;
    uint32_t* dIndices = static_cast<uint32_t*>(memory) + totalVertexDataCount * 8;

    CUDA_CHECK(cudaMemcpy2D(dPositionsU, sizeof(Vector4f),
                            postitionsCPU.data(), sizeof(float) * 3,
                            sizeof(float) * 3, totalVertexDataCount,
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dNormalsV, sizeof(Vector4f),
                            normalsCPU.data(), sizeof(float) * 3,
                            sizeof(float) * 3, totalVertexDataCount,
                            cudaMemcpyHostToDevice));
    // Strided Copy of UVs
    CUDA_CHECK(cudaMemcpy2D(dPositionsU + 3, sizeof(Vector4f),
                            uvsCPU.data(), sizeof(float) * 2,
                            sizeof(float), totalVertexDataCount,
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dNormalsV + 3, sizeof(Vector4f),
                            uvsCPU.data() + 1, sizeof(float) * 2,
                            sizeof(float), totalVertexDataCount,
                            cudaMemcpyHostToDevice));
    // Copy Indices
    CUDA_CHECK(cudaMemcpy(dIndices, indexCPU.data(),
                          totalVertexCount * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Set Main Pointers of batch
    dData.positionsU = reinterpret_cast<Vector4f*>(dPositionsU);
    dData.normalsV = reinterpret_cast<Vector4f*>(dNormalsV);
    dData.indexList = dIndices;
    return e;
}

SceneError GPUPrimitiveTriangle::ChangeTime(const NodeListing& surfaceDataNodes, double time,
                                            const SurfaceLoaderGeneratorI& loaderGen)
{
    SceneError e = SceneError::OK;
    std::vector<std::vector<size_t>> primCountList;
    std::vector<std::vector<size_t>> offsetList;

    // Generate Loaders
    std::vector<SharedLibPtr<SurfaceLoaderI>> loaders;
    for(const auto& sPtr : surfaceDataNodes)
    {
        const SceneNodeI& s = *sPtr;
        SharedLibPtr<SurfaceLoaderI> sl(nullptr, [](SurfaceLoaderI*)->void{});
        if((e = loaderGen.GenerateSurfaceLoader(sl, s, time)) != SceneError::OK)
            return e;
        loaders.emplace_back(std::move(sl));
    }

    // First update aabbs (these should have changed)
    totalPrimitiveCount = 0;
    for(const auto& loader : loaders)
    {
        const SceneNodeI& node = loader->SceneNode();
        const size_t batchCount = node.IdCount();

        std::vector<AABB3>  aabbList(batchCount);
        primCountList.emplace_back(batchCount);
        offsetList.emplace_back(batchCount + 1);

        // Load Aux Data
        if((e = loader->PrimitiveCounts(primCountList.back().data())) != SceneError::OK)
            return e;
        if((e = loader->AABB(aabbList.data())) != SceneError::OK)
            return e;
        if((e = loader->BatchOffsets(offsetList.back().data())) != SceneError::OK)
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
    size_t j = 0;
    std::vector<float> postitionsCPU, normalsCPU, uvsCPU;
    for(const auto& loader : loaders)
    {
        const SceneNodeI& node = loader->SceneNode();
        const size_t batchCount = node.IdCount();

        const std::vector<size_t>& offsets = offsetList[j];
        const std::vector<size_t>& counts = primCountList[j];
        size_t loaderTotalCount = offsets.back();
        size_t loaderTotalVertexCount = loaderTotalCount * 3;

        postitionsCPU.resize(loaderTotalVertexCount * 3);
        normalsCPU.resize(loaderTotalVertexCount * 2);
        uvsCPU.resize(loaderTotalVertexCount * 2);
        
        if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(postitionsCPU.data()),
                                         PrimitiveDataType::POSITION)) != SceneError::OK)
            return e;
        if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(normalsCPU.data()),
                                         PrimitiveDataType::NORMAL)) != SceneError::OK)
            return e;
        if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(uvsCPU.data()),
                                         PrimitiveDataType::UV)) != SceneError::OK)
            return e;

        // Now copy one by one
        size_t i = 0;
        for(const auto& pair : node.Ids())
        {
            NodeId id = pair.first;
            Vector2ul range = batchDataRanges[id];

            size_t primitiveCount = counts[i];
            assert((range[1] - range[0]) == primitiveCount);

            float* dPositionsU = static_cast<float*>(memory);
            float* dNormalsV = static_cast<float*>(memory) + totalPrimitiveCount;

            // Pos and Normal
            CUDA_CHECK(cudaMemcpy2D(dPositionsU + range[0], sizeof(Vector4f),
                                    postitionsCPU.data(), sizeof(float) * 3,
                                    sizeof(float) * 3, primitiveCount,
                                    cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy2D(dNormalsV + range[0], sizeof(Vector4f),
                                    normalsCPU.data(), sizeof(float) * 3,
                                    sizeof(float) * 3, primitiveCount,
                                    cudaMemcpyHostToDevice));
            // Strided Copy of UVs
            CUDA_CHECK(cudaMemcpy2D(dPositionsU + range[0] + 3, sizeof(Vector4f),
                                    uvsCPU.data() + offsets[i], sizeof(float) * 2,
                                    sizeof(float), primitiveCount,
                                    cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy2D(dNormalsV + range[0] + 3, sizeof(Vector4f),
                                    uvsCPU.data() + offsets[i] + 1, sizeof(float) * 2,
                                    sizeof(float), primitiveCount,
                                    cudaMemcpyHostToDevice));
        }
        j++;
    }
    return e;
}

Vector2ul GPUPrimitiveTriangle::PrimitiveBatchRange(uint32_t surfaceDataId) const
{
    return batchRanges.at(surfaceDataId);
}

AABB3 GPUPrimitiveTriangle::PrimitiveBatchAABB(uint32_t surfaceDataId) const
{
    return batchAABBs.at(surfaceDataId);
}

bool GPUPrimitiveTriangle::CanGenerateData(const std::string& s) const
{
    return (s == PrimitiveDataTypeToString(PrimitiveDataType::POSITION) ||
            s == PrimitiveDataTypeToString(PrimitiveDataType::NORMAL) ||
            s == PrimitiveDataTypeToString(PrimitiveDataType::UV));
}