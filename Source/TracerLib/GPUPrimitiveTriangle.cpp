#include "GPUPrimitiveTriangle.h"

#include "RayLib/SceneError.h"
#include "RayLib/Log.h"

#include "RayLib/SurfaceLoaderGenerator.h"
#include "RayLib/SurfaceLoaderI.h"

#include <execution>

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
                                                 const SurfaceLoaderGeneratorI& loaderGen,
                                                 const std::string& scenePath)
{
    SceneError e = SceneError::OK;
    std::vector<size_t> loaderVOffsets, loaderIOffsets;

    // Generate Loaders
    std::vector<SharedLibPtr<SurfaceLoaderI>> loaders;
    for(const auto& sPtr : surfaceDataNodes)
    {
        const SceneNodeI& s = *sPtr;
        SharedLibPtr<SurfaceLoaderI> sl(nullptr, [] (SurfaceLoaderI*)->void {});        
        if((e = loaderGen.GenerateSurfaceLoader(sl, scenePath, s, time)) != SceneError::OK)
           return e;

        try
        {
            loaders.emplace_back(std::move(sl));
        }
        catch(SceneException const& e)
        {
            if(e.what()[0] != '\0') METU_ERROR_LOG("%s", e.what());
            return e;
        }
    }

    // Do sanity check for data type matching
    for(const auto& loader : loaders)
    {
        PrimitiveDataLayout positionLayout, uvLayout, normalLayout, indexLayout;
        if((e = loader->PrimDataLayout(positionLayout, PrimitiveDataType::POSITION)) != SceneError::OK)
            return e;
        if((e = loader->PrimDataLayout(uvLayout, PrimitiveDataType::UV)) != SceneError::OK)
            return e;
        if((e = loader->PrimDataLayout(normalLayout, PrimitiveDataType::NORMAL)) != SceneError::OK)
            return e;
        if((e = loader->PrimDataLayout(indexLayout, PrimitiveDataType::VERTEX_INDEX)) != SceneError::OK)
            return e;

        // TODO: Add 32 bit index support here (to the entire function as well
        if(positionLayout != POS_LAYOUT || uvLayout != UV_LAYOUT ||
           normalLayout != NORMAL_LAYOUT || indexLayout != INDEX_LAYOUT)
            return SceneError::SURFACE_DATA_PRIMITIVE_MISMATCH;
    }

    totalDataCount = 0;    
    totalPrimitiveCount = 0;
    size_t totalIndexCount = 0;
    for(const auto& loader : loaders)
    {
        const SceneNodeI& node = loader->SceneNode();

        std::vector<AABB3> aabbList;
        std::vector<Vector2ul> primRange;
        std::vector<Vector2ul> dataRange;
        //std::vector<size_t> primCounts;
        size_t loaderPCount, loaderUVCount, loaderNCount, loaderICount;

        // Load Aux Data
        if((e = loader->AABB(aabbList)) != SceneError::OK)
            return e;
        if((e = loader->PrimitiveRanges(primRange)) != SceneError::OK)
            return e;
        //if((e = loader->PrimitiveCounts(primCounts)) != SceneError::OK)
        //    return e;
        if((e = loader->PrimitiveDataRanges(dataRange)) != SceneError::OK)
            return e;
        if((e = loader->PrimitiveDataCount(loaderPCount, PrimitiveDataType::POSITION)) != SceneError::OK)
            return e;
        if((e = loader->PrimitiveDataCount(loaderUVCount, PrimitiveDataType::UV)) != SceneError::OK)
            return e;
        if((e = loader->PrimitiveDataCount(loaderNCount, PrimitiveDataType::NORMAL)) != SceneError::OK)
            return e;
        if((e = loader->PrimitiveDataCount(loaderICount, PrimitiveDataType::VERTEX_INDEX)) != SceneError::OK)
            return e;

        // Single indexed vertex data sanity check
        assert(loaderPCount == loaderUVCount &&
               loaderUVCount == loaderNCount);

        // Populate
        size_t i = 0, totalPrimCountOnLoader = 0;
        for(const auto& pair : node.Ids())
        {
            NodeId id = pair.first;

            Vector2ul currentPRange = primRange[i];
            currentPRange += Vector2ul(totalPrimitiveCount);

            Vector2ul currentDRange = dataRange[i];
            currentDRange += Vector2ul(totalDataCount);

            batchRanges.emplace(id, currentPRange);
            batchDataRanges.emplace(id, currentDRange);
            batchAABBs.emplace(id, aabbList[i]);

            totalPrimCountOnLoader += primRange[i][1] - primRange[i][0];
            i++;
        }

        // Save start offsets for each loader (for data copy)
        loaderVOffsets.emplace_back(totalDataCount);
        loaderIOffsets.emplace_back(totalIndexCount);

        totalPrimitiveCount += totalPrimCountOnLoader;
        totalDataCount += loaderPCount;
        totalIndexCount += loaderICount;
    }
    assert(totalPrimitiveCount * 3 == totalIndexCount);
    loaderVOffsets.emplace_back(totalDataCount);
    loaderIOffsets.emplace_back(totalIndexCount);

    // Now allocate to CPU then GPU
    constexpr size_t VertPosSize = PrimitiveDataLayoutToSize(POS_LAYOUT);
    constexpr size_t VertUVSize = PrimitiveDataLayoutToSize(UV_LAYOUT);
    constexpr size_t VertNormSize = PrimitiveDataLayoutToSize(NORMAL_LAYOUT);
    constexpr size_t IndexSize = PrimitiveDataLayoutToSize(INDEX_LAYOUT);
    std::vector<Byte> postitionsCPU(totalDataCount * VertPosSize);
    std::vector<Byte> uvsCPU(totalDataCount* VertUVSize);
    std::vector<Byte> normalsCPU(totalDataCount * VertNormSize);
    std::vector<Byte> indexCPU(totalIndexCount * IndexSize);

    size_t i = 0;
    for(const auto& loader : loaders)
    {
        const SceneNodeI& node = loader->SceneNode();

        const size_t offsetVertex = loaderVOffsets[i];
        const size_t offsetIndex = loaderIOffsets[i];
        const size_t offsetIndexNext = loaderIOffsets[i + 1];

        if((e = loader->GetPrimitiveData(postitionsCPU.data() + offsetVertex * VertPosSize,
                                         PrimitiveDataType::POSITION)) != SceneError::OK)
            return e;
        if((e = loader->GetPrimitiveData(uvsCPU.data() + offsetVertex * VertUVSize,
                                         PrimitiveDataType::UV)) != SceneError::OK)
            return e;
        if((e = loader->GetPrimitiveData(normalsCPU.data() + offsetVertex * VertNormSize,
                                         PrimitiveDataType::NORMAL)) != SceneError::OK)
            return e;
        if((e = loader->GetPrimitiveData(indexCPU.data() + offsetIndex * IndexSize,
                                         PrimitiveDataType::VERTEX_INDEX)) != SceneError::OK)
            return e;

        // Accumulate the offset to the indices
        // TODO: utilize GPU here maybe
        if(i != 0)
        {
            std::for_each(std::execution::par_unseq,
                          reinterpret_cast<uint64_t*>(indexCPU.data() + offsetIndex * IndexSize),
                          reinterpret_cast<uint64_t*>(indexCPU.data() + offsetIndexNext * IndexSize),
                          [&](uint64_t& t) { t += offsetVertex; });
        }
        i++;
    }

    std::vector<uint64_t>asd(totalIndexCount);
    std::copy(reinterpret_cast<uint64_t*>(indexCPU.data()),
              reinterpret_cast<uint64_t*>(indexCPU.data()) + totalIndexCount,
              asd.data());

    // All loaded to CPU, copy to GPU
    // Alloc
    memory = std::move(DeviceMemory(sizeof(Vector4f) * 2 * totalDataCount +
                                    sizeof(uint64_t) * totalIndexCount));
    Byte* dPositionsU = static_cast<Byte*>(memory);
    Byte* dNormalsV = static_cast<Byte*>(memory) + totalDataCount * sizeof(Vector4f);
    Byte* dIndices = static_cast<Byte*>(memory) + totalDataCount * sizeof(Vector4f) * 2;

    CUDA_CHECK(cudaMemcpy2D(dPositionsU, sizeof(Vector4f),
                            postitionsCPU.data(), sizeof(float) * 3,
                            sizeof(float) * 3, totalDataCount,
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dNormalsV, sizeof(Vector4f),
                            normalsCPU.data(), sizeof(float) * 3,
                            sizeof(float) * 3, totalDataCount,
                            cudaMemcpyHostToDevice));
    // Strided Copy of UVs
    CUDA_CHECK(cudaMemcpy2D(dPositionsU + VertPosSize, sizeof(Vector4f),
                            uvsCPU.data(), sizeof(float) * 2,
                            sizeof(float), totalDataCount,
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dNormalsV + VertNormSize, sizeof(Vector4f),
                            uvsCPU.data() + 1, sizeof(float) * 2,
                            sizeof(float), totalDataCount,
                            cudaMemcpyHostToDevice));
    // Copy Indices
    CUDA_CHECK(cudaMemcpy(dIndices, indexCPU.data(),
                          totalIndexCount * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    // Set Main Pointers of batch
    dData.positionsU = reinterpret_cast<Vector4f*>(dPositionsU);
    dData.normalsV = reinterpret_cast<Vector4f*>(dNormalsV);
    dData.indexList = reinterpret_cast<uint64_t*>(dIndices);
    return e;
}

SceneError GPUPrimitiveTriangle::ChangeTime(const NodeListing& surfaceDataNodes, double time,
                                            const SurfaceLoaderGeneratorI& loaderGen,
                                            const std::string& scenePath)
{
    return SceneError::OK;
    //SceneError e = SceneError::OK;
    //std::vector<std::vector<size_t>> primCountList;
    //std::vector<std::vector<size_t>> offsetList;

    //// Generate Loaders
    //std::vector<SharedLibPtr<SurfaceLoaderI>> loaders;
    //for(const auto& sPtr : surfaceDataNodes)
    //{
    //    const SceneNodeI& s = *sPtr;
    //    SharedLibPtr<SurfaceLoaderI> sl(nullptr, [](SurfaceLoaderI*)->void{});
    //    if((e = loaderGen.GenerateSurfaceLoader(sl, s, time)) != SceneError::OK)
    //        return e;
    //    loaders.emplace_back(std::move(sl));
    //}

    //// First update aabbs (these should have changed)
    //totalPrimitiveCount = 0;
    //for(const auto& loader : loaders)
    //{
    //    const SceneNodeI& node = loader->SceneNode();
    //    const size_t batchCount = node.IdCount();

    //    std::vector<AABB3>  aabbList(batchCount);
    //    primCountList.emplace_back(batchCount);
    //    offsetList.emplace_back(batchCount + 1);

    //    // Load Aux Data
    //    if((e = loader->PrimitiveCounts(primCountList.back().data())) != SceneError::OK)
    //        return e;
    //    if((e = loader->AABB(aabbList.data())) != SceneError::OK)
    //        return e;
    //    if((e = loader->BatchOffsets(offsetList.back().data())) != SceneError::OK)
    //        return e;

    //    size_t i = 0;
    //    for(const auto& pair : node.Ids())
    //    {
    //        NodeId id = pair.first;
    //        const Vector2ul range = batchRanges.at(id);
    //        assert((range[1] - range[0]) == primCountList.back()[i]);
    //        batchAABBs.at(id) = aabbList[i];

    //        i++;
    //    }
    //}

    //// Now Copy
    //size_t j = 0;
    //std::vector<float> postitionsCPU, normalsCPU, uvsCPU;
    //for(const auto& loader : loaders)
    //{
    //    const SceneNodeI& node = loader->SceneNode();
    //    const size_t batchCount = node.IdCount();

    //    const std::vector<size_t>& offsets = offsetList[j];
    //    const std::vector<size_t>& counts = primCountList[j];
    //    size_t loaderTotalCount = offsets.back();
    //    size_t loaderTotalVertexCount = loaderTotalCount * 3;

    //    postitionsCPU.resize(loaderTotalVertexCount * 3);
    //    normalsCPU.resize(loaderTotalVertexCount * 2);
    //    uvsCPU.resize(loaderTotalVertexCount * 2);
    //    
    //    if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(postitionsCPU.data()),
    //                                     PrimitiveDataType::POSITION)) != SceneError::OK)
    //        return e;
    //    if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(normalsCPU.data()),
    //                                     PrimitiveDataType::NORMAL)) != SceneError::OK)
    //        return e;
    //    if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(uvsCPU.data()),
    //                                     PrimitiveDataType::UV)) != SceneError::OK)
    //        return e;

    //    // Now copy one by one
    //    size_t i = 0;
    //    for(const auto& pair : node.Ids())
    //    {
    //        NodeId id = pair.first;
    //        Vector2ul range = batchDataRanges[id];

    //        size_t primitiveCount = counts[i];
    //        assert((range[1] - range[0]) == primitiveCount);

    //        Byte* dPositionsU = static_cast<Byte*>(memory);
    //        Byte* dNormalsV = static_cast<Byte*>(memory) + totalPrimitiveCount;

    //        size_t rangeOffset = range[0] * sizeof(Vector4f);

    //        // Pos and Normal
    //        CUDA_CHECK(cudaMemcpy2D(dPositionsU + rangeOffset, sizeof(Vector4f),
    //                                postitionsCPU.data(), sizeof(float) * 3,
    //                                sizeof(float) * 3, primitiveCount,
    //                                cudaMemcpyHostToDevice));
    //        CUDA_CHECK(cudaMemcpy2D(dNormalsV + rangeOffset, sizeof(Vector4f),
    //                                normalsCPU.data(), sizeof(float) * 3,
    //                                sizeof(float) * 3, primitiveCount,
    //                                cudaMemcpyHostToDevice));
    //        // Strided Copy of UVs
    //        CUDA_CHECK(cudaMemcpy2D(dPositionsU + range[0] + 3, sizeof(Vector4f),
    //                                uvsCPU.data() + offsets[i], sizeof(float) * 2,
    //                                sizeof(float), primitiveCount,
    //                                cudaMemcpyHostToDevice));
    //        CUDA_CHECK(cudaMemcpy2D(dNormalsV + range[0] + 3, sizeof(Vector4f),
    //                                uvsCPU.data() + offsets[i] + 1, sizeof(float) * 2,
    //                                sizeof(float), primitiveCount,
    //                                cudaMemcpyHostToDevice));
    //    }
    //    j++;
    //}
    //return e;
}

bool GPUPrimitiveTriangle::HasPrimitive(uint32_t surfaceDataId) const
{
    auto it = batchRanges.end();
    if((it = batchRanges.find(surfaceDataId)) == batchRanges.end())
       return false;
    return true;
}

SceneError GPUPrimitiveTriangle::GenerateLights(std::vector<LightStruct>& result,
                                                const Vector3& flux, HitKey key,
                                                uint32_t id) const
{
    const auto& range = batchRanges.at(id);
    result.reserve(range[1] - range[0]);
    for(uint64_t i = range[0]; i < range[1]; i++)
    {
        LightStruct ls;
        ls.matKey = key;
        ls.flux = flux;
        ls.type = LightType::TRIANGULAR;
        ls.position0 = dData.positionsU[dData.indexList[i * 3 + 0]];
        ls.position1 = dData.positionsU[dData.indexList[i * 3 + 1]];
        ls.position2 = dData.positionsU[dData.indexList[i * 3 + 2]];

        result.push_back(ls);
    }
    return SceneError::OK;
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