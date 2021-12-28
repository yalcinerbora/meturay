#include "GPUPrimitiveSphere.h"

#include "RayLib/SceneError.h"
#include "RayLib/SceneNodeI.h"
#include "RayLib/SurfaceLoaderGenerator.h"
#include "RayLib/SurfaceLoaderI.h"
#include "RayLib/Log.h"

// Generics
GPUPrimitiveSphere::GPUPrimitiveSphere()
    : totalPrimitiveCount(0)
{}

const char* GPUPrimitiveSphere::Type() const
{
    return TypeName();
}

SceneError GPUPrimitiveSphere::InitializeGroup(const NodeListing& surfaceDataNodes,
                                               double time,
                                               const SurfaceLoaderGeneratorI& loaderGen,
                                               const TextureNodeMap&,
                                               const std::string& scenePath)
{
    SceneError e = SceneError::OK;
    std::vector<size_t> loaderOffsets;

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
            if(e.what()[0] != '\0') METU_ERROR_LOG("{:s}", std::string(e.what()));
            return e;
        }
    }

    // Do sanity check for data type matching
    for(const auto& loader : loaders)
    {
        PrimitiveDataLayout positionLayout, radiusLayout;
        if((e = loader->PrimDataLayout(positionLayout, PrimitiveDataType::POSITION)) != SceneError::OK)
            return e;
        if((e = loader->PrimDataLayout(radiusLayout, PrimitiveDataType::RADIUS)) != SceneError::OK)
            return e;

        if(positionLayout != POS_LAYOUT || radiusLayout != RADUIS_LAYOUT)
            return SceneError::SURFACE_DATA_PRIMITIVE_MISMATCH;
    }

    totalPrimitiveCount = 0;
    for(const auto& loader : loaders)
    {
        const SceneNodeI& node = loader->SceneNode();

        std::vector<AABB3> aabbList;
        std::vector<Vector2ul> primRange;
        std::vector<Vector2ul> dataRange;
        //std::vector<size_t> primCounts;
        size_t loaderPCount, loaderRCount;

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
        if((e = loader->PrimitiveDataCount(loaderRCount, PrimitiveDataType::RADIUS)) != SceneError::OK)
            return e;

        // Single indexed vertex data sanity check
        assert(loaderPCount == loaderRCount);

        // Populate
        size_t i = 0, totalPrimCountOnLoader = 0;
        for(const auto& pair : node.Ids())
        {
            NodeId id = pair.first;

            Vector2ul currentPRange = primRange[i];
            currentPRange += Vector2ul(totalPrimitiveCount);

            batchRanges.emplace(id, currentPRange);
            batchAABBs.emplace(id, aabbList[i]);

            totalPrimCountOnLoader += (primRange[i][1] - primRange[i][0]);
            i++;
        }

        // Save start offsets for each loader (for data copy)
        loaderOffsets.emplace_back(totalPrimitiveCount);
        totalPrimitiveCount += totalPrimCountOnLoader;
    }
    loaderOffsets.emplace_back(totalPrimitiveCount);
    // Now allocate to CPU then GPU
    constexpr size_t SphrPosSize = PrimitiveDataLayoutToSize(POS_LAYOUT);
    constexpr size_t SphrRadSize = PrimitiveDataLayoutToSize(RADUIS_LAYOUT);
    std::vector<Byte> postitionsCPU(totalPrimitiveCount * SphrPosSize);
    std::vector<Byte> radiusCPU(totalPrimitiveCount * SphrRadSize);

    size_t i = 0;
    for(const auto& loader : loaders)
    {
        //const SceneNodeI& node = loader->SceneNode();
        const size_t offset = loaderOffsets[i];

        // Load Data in Batch
        if((e = loader->GetPrimitiveData(postitionsCPU.data() + offset * SphrPosSize,
                                         PrimitiveDataType::POSITION)) != SceneError::OK)
            return e;
        if((e = loader->GetPrimitiveData(radiusCPU.data() + offset * SphrRadSize,
                                         PrimitiveDataType::RADIUS)) != SceneError::OK)
            return e;
        i++;
    }

    // All loaded to CPU, copy to GPU
    // Alloc
    memory = DeviceMemory(sizeof(Vector4f) * totalPrimitiveCount);
    float* dCentersRadius = static_cast<float*>(memory);
    // Copy
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

SceneError GPUPrimitiveSphere::ChangeTime(const NodeListing&, double,
                                          const SurfaceLoaderGeneratorI&,
                                          const std::string&)
{
    // TODO: Implement
    return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;
}

Vector2ul GPUPrimitiveSphere::PrimitiveBatchRange(uint32_t surfaceDataId) const
{
    return batchRanges.at(surfaceDataId);
}

AABB3 GPUPrimitiveSphere::PrimitiveBatchAABB(uint32_t surfaceDataId) const
{
    return batchAABBs.at(surfaceDataId);
}

bool GPUPrimitiveSphere::PrimitiveBatchHasAlphaMap(uint32_t) const
{
    // TODO: add alpha map support for sphere as well
    return false;
}

bool GPUPrimitiveSphere::PrimitiveBatchBackFaceCulled(uint32_t) const
{
    // De don't do back face culling on sphere
    return false;
}

uint64_t GPUPrimitiveSphere::TotalPrimitiveCount() const
{
    return totalPrimitiveCount;
}

uint64_t GPUPrimitiveSphere::TotalDataCount() const
{
    return totalPrimitiveCount;
}

bool GPUPrimitiveSphere::CanGenerateData(const std::string& s) const
{
    return (s == PrimitiveDataTypeToString(PrimitiveDataType::POSITION) ||
            s == PrimitiveDataTypeToString(PrimitiveDataType::NORMAL) ||
            s == PrimitiveDataTypeToString(PrimitiveDataType::UV));
}