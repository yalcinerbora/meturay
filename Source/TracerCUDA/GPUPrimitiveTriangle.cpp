#include "GPUPrimitiveTriangle.h"

#include "RayLib/SceneError.h"
#include "RayLib/Log.h"
#include "RayLib/SurfaceLoaderGenerator.h"
#include "RayLib/SurfaceLoaderI.h"
#include "RayLib/MemoryAlignment.h"

//#include "TracerDebug.h"
#include "TextureFunctions.h"

#include <execution>
#include <ranges>

struct IndexTriplet
{
    uint64_t i[3];

    uint64_t& operator[](int j) { return i[j]; }
    const uint64_t& operator[](int j) const { return i[j]; }
};
static_assert(sizeof(IndexTriplet) == sizeof(uint64_t) * 3);

//std::ostream& operator<<(std::ostream& stream, const IndexTriplet& t)
//{
//    stream << t[0] << ", " << t[1] << ", " << t[2];
//    return stream;
//}

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
                                                 const TextureNodeMap& textureNodes,
                                                 const std::string& scenePath)
{
    SceneError e = SceneError::OK;
    std::vector<size_t> loaderVOffsets, loaderIOffsets;

    std::vector<uint64_t> batchOffsets;
    std::vector<Byte> cullFaceFlags;
    OptionalNodeList<NodeTextureStruct> alphaMapInfo;

    // Generate Loaders
    std::vector<SharedLibPtr<SurfaceLoaderI>> loaders;
    for(const auto& sPtr : surfaceDataNodes)
    {
        const SceneNodeI& s = *sPtr;
        SharedLibPtr<SurfaceLoaderI> sl(nullptr, [](SurfaceLoaderI*)->void {});
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
        // Check alpha maps
        auto alphaMaps = s.AccessOptionalTextureNode(ALPHA_MAP_NAME, time);
        alphaMapInfo.insert(alphaMapInfo.end(), alphaMaps.cbegin(), alphaMaps.cend());
        // Check optional cull flag
        std::vector<bool> cullData;
        if(s.CheckNode(CULL_FLAG_NAME))
        {
            cullData = s.AccessBool(CULL_FLAG_NAME);
        }
        else cullData.resize(s.IdCount(), true);
        cullFaceFlags.insert(cullFaceFlags.end(), cullData.cbegin(), cullData.cend());

        // Populate HasAlphaMap & Cull Face flags for each prim batch
        int i = 0;
        for(const auto& pair : s.Ids())
        {
            NodeId id = pair.first;
            batchAlphaMapFlag.emplace(id, alphaMaps[i].first);
            batchBackFaceCullFlag.emplace(id, static_cast<bool>(cullData[i]));
            i++;
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
        size_t loaderPCount, loaderUVCount, loaderICount;

        // Load Aux Data
        if((e = loader->AABB(aabbList)) != SceneError::OK)
            return e;
        if((e = loader->PrimitiveRanges(primRange)) != SceneError::OK)
            return e;
        if((e = loader->PrimitiveDataCount(loaderPCount, PrimitiveDataType::POSITION)) != SceneError::OK)
            return e;
        if((e = loader->PrimitiveDataCount(loaderUVCount, PrimitiveDataType::UV)) != SceneError::OK)
            return e;
        if((e = loader->PrimitiveDataCount(loaderICount, PrimitiveDataType::VERTEX_INDEX)) != SceneError::OK)
            return e;

        // Single indexed vertex data sanity check
        assert(loaderPCount == loaderUVCount);

        // Populate
        size_t i = 0, totalPrimCountOnLoader = 0;
        for(const auto& pair : node.Ids())
        {
            NodeId id = pair.first;

            Vector2ul currentPRange = primRange[i];
            currentPRange += Vector2ul(totalPrimitiveCount);

            //Vector2ul currentDRange = dataRange[i];
            //currentDRange += Vector2ul(totalDataCount);
            //batchDataRanges.emplace(id, currentDRange);

            batchRanges.emplace(id, currentPRange);
            batchAABBs.emplace(id, aabbList[i]);
            batchOffsets.push_back(currentPRange[0]);

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
    batchOffsets.push_back(totalPrimitiveCount);

    // Now allocate to CPU then GPU
    constexpr size_t VertPosSize = PrimitiveDataLayoutToSize(POS_LAYOUT);
    constexpr size_t VertUVSize = PrimitiveDataLayoutToSize(UV_LAYOUT);
    //constexpr size_t VertTangentSize = PrimitiveDataLayoutToSize(TANGENT_LAYOUT);
    //constexpr size_t VertNormSize = PrimitiveDataLayoutToSize(NORMAL_LAYOUT);
    constexpr size_t IndexSize = PrimitiveDataLayoutToSize(INDEX_LAYOUT);
    constexpr size_t RotationSize = sizeof(QuatF);

    // Stationary buffers
    std::vector<Byte> postitionsCPU(totalDataCount * VertPosSize);
    std::vector<Byte> uvsCPU(totalDataCount * VertUVSize);
    std::vector<Byte> rotationsCPU(totalDataCount * RotationSize);
    std::vector<Byte> indexCPU(totalIndexCount * IndexSize);

    // Temporary buffers (re-allocated per batch)
    std::vector<Vector3> normals;

    size_t i = 0;
    for(const auto& loader : loaders)
    {
        const size_t offsetVertex = loaderVOffsets[i];
        const size_t offsetIndex = loaderIOffsets[i];
        const size_t offsetIndexNext = loaderIOffsets[i + 1];
        const size_t vertexCount = loaderVOffsets[i + 1] - offsetVertex;

        // Load Mandatory Data
        if((e = loader->GetPrimitiveData(postitionsCPU.data() + offsetVertex * VertPosSize,
                                         PrimitiveDataType::POSITION)) != SceneError::OK)
            return e;
        if((e = loader->GetPrimitiveData(uvsCPU.data() + offsetVertex * VertUVSize,
                                         PrimitiveDataType::UV)) != SceneError::OK)
            return e;
        if((e = loader->GetPrimitiveData(indexCPU.data() + offsetIndex * IndexSize,
                                         PrimitiveDataType::VERTEX_INDEX)) != SceneError::OK)
            return e;

        // Get temporary data to generate tangent space transformation
        normals.resize(vertexCount);
        if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(normals.data()),
                                         PrimitiveDataType::NORMAL)) != SceneError::OK)
            return e;

        // Check if tangents are available
        bool hasTangent;
        if((e = loader->HasPrimitiveData(hasTangent, PrimitiveDataType::TANGENT)) != SceneError::OK)
            return e;

        // Access index three by three for data generation
        IndexTriplet* primStart = reinterpret_cast<IndexTriplet*>(indexCPU.data() + offsetIndex * IndexSize);
        IndexTriplet* primEnd = reinterpret_cast<IndexTriplet*>(indexCPU.data() + offsetIndexNext * IndexSize);

        // Manipulate Ptrs of in/out
        QuatF* rotationsOut = reinterpret_cast<QuatF*>(rotationsCPU.data() + offsetVertex * RotationSize);
        const Vector3* normalsIn = normals.data();

        // Allocate Array of Vector3 atomics
        std::vector<Vector3f> tangents(vertexCount, Zero3f);
        // If object does not come with per-vertex tangents
        // Calculate and average the values
        if(!hasTangent)
        {
            const Vector3* positionsIn = reinterpret_cast<Vector3*>(postitionsCPU.data() + offsetVertex * VertPosSize);
            const Vector2* uvsIn = reinterpret_cast<Vector2*>(uvsCPU.data() + offsetVertex * VertUVSize);

            // Utilize position and uv for quat generation
            std::for_each(primStart, primEnd,
                          [&](IndexTriplet& indices)
                          {
                              using namespace Triangle;

                              Vector3 normals[3];
                              normals[0] = normalsIn[indices[0]];
                              normals[1] = normalsIn[indices[1]];
                              normals[2] = normalsIn[indices[2]];

                              Vector3 pos[3];
                              pos[0] = positionsIn[indices[0]];
                              pos[1] = positionsIn[indices[1]];
                              pos[2] = positionsIn[indices[2]];

                              Vector2 uvs[3];
                              uvs[0] = uvsIn[indices[0]];
                              uvs[1] = uvsIn[indices[1]];
                              uvs[2] = uvsIn[indices[2]];

                              // Generate the tangents for this triangle orientation
                              Vector3f t0 = CalculateTangent(pos[0], pos[1], pos[2],
                                                             uvs[0], uvs[1], uvs[2],
                                                             normals[0]);
                              Vector3f t1 = CalculateTangent(pos[1], pos[2], pos[0],
                                                             uvs[1], uvs[2], uvs[0],
                                                             normals[1]);
                              Vector3f t2 = CalculateTangent(pos[2], pos[0], pos[1],
                                                             uvs[2], uvs[0], uvs[1],
                                                             normals[2]);

                              // Degenerate triangle is found,
                              // (or uv's are degenerate)
                              // arbitrarily find a tangent
                              if(t0.HasNaN()) t0 = OrthogonalVector(normals[0]);
                              if(t1.HasNaN()) t1 = OrthogonalVector(normals[1]);
                              if(t2.HasNaN()) t2 = OrthogonalVector(normals[2]);

                              tangents[indices[0]] += t0;
                              tangents[indices[1]] += t1;
                              tangents[indices[2]] += t2;
                          });
        }
        else
        {
            // Primitive already have tangents just load it
            size_t tangentCount;
            if((e = loader->PrimitiveDataCount(tangentCount, PrimitiveDataType::TANGENT)) != SceneError::OK)
                return e;

            if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(tangents.data()),
                                             PrimitiveDataType::TANGENT)) != SceneError::OK)
                return e;
        }

        // Generated (or loaded) the tangents
        // Generate tangent space rotations
        const Vector3* tangentsIn = tangents.data();

        // Utilize tangent and normal for quat generation
        std::ranges::iota_view vertIndices(size_t(0), vertexCount);
        std::for_each(vertIndices.begin(),
                      vertIndices.end(),
                      [&](size_t index)
                      {
                          Vector3 n = normalsIn[index];
                          Vector3 t = tangentsIn[index];

                          // Neighboring vertices of the tangents
                          // are canceled each other out
                          // Just find an arbitrary tangent
                          if(t.LengthSqr() < MathConstants::Epsilon)
                              t = OrthogonalVector(n);

                          // If tangents are generated
                          // each triangle that shared this vertices
                          // added its tangent, normalize it
                          t.NormalizeSelf();

                          // Gram-Schmidt orthonormalization
                          // This is required since normal may be skewed to hide
                          // edges (to create smooth lighting)
                          Vector3f tNorm = (t - n * n.Dot(t)).Normalize();
                          // Bitangent
                          Vector3f b = Cross(n, tNorm);

                          // Generate rotation
                          QuatF q;
                          TransformGen::Space(q, tNorm, b, n);

                          rotationsOut[index] = q;
                      });

        // Offset the indices with respect to the GlobalPrimitive Buffer
        // Don't offset for the very first mesh
        if(i != 0)
        {
            std::for_each(std::execution::par_unseq,
                          primStart, primEnd,
                          [&](IndexTriplet& indices)
                          {
                              indices[0] += offsetVertex;
                              indices[1] += offsetVertex;
                              indices[2] += offsetVertex;
                          });
        }
        i++;
    }

    // Construct Alpha Maps
    std::vector<uint32_t> bitMapIndex;
    std::vector<Vector2ui> bitmapDims;
    std::vector<std::vector<Byte>> bitmapData;
    bitMapIndex.reserve(alphaMapInfo.size());
    for(const auto& texInfo : alphaMapInfo)
    {
        if(!texInfo.first)
        {
            bitMapIndex.push_back(std::numeric_limits<uint32_t>::max());
            continue;
        }

        uint32_t textureId = texInfo.second.texId;
        TextureAccessLayout access = texInfo.second.channelLayout;

        if(access != TextureAccessLayout::R &&
           access != TextureAccessLayout::G &&
           access != TextureAccessLayout::B &&
           access != TextureAccessLayout::A)
            return SceneError::BITMAP_LOAD_CALLED_WITH_MULTIPLE_CHANNELS;

        TextureChannelType channel = TextureFunctions::TextureAccessLayoutToTextureChannels(access)[0];

        // Check if this texture/channel pair is already loaded
        auto loc = loadedBitmaps.cend();
        if((loc = loadedBitmaps.find(std::make_pair(textureId, channel))) != loadedBitmaps.cend())
        {
            bitMapIndex.push_back(loc->second);
        }
        else
        {
            Vector2ui dim;
            std::vector<Byte> bm;
            if((e = TextureFunctions::LoadBitMap(bm,
                                                 dim,
                                                 textureId,
                                                 channel,
                                                 textureNodes,
                                                 scenePath)) != SceneError::OK)
                return e;

            bitMapIndex.push_back(static_cast<uint32_t>(bitmapData.size()));
            bitmapData.emplace_back(std::move(bm));
            bitmapDims.emplace_back(std::move(dim));
        }
    }
    assert(bitmapData.size() == bitmapDims.size());

    // Generate Bitmap Group
    bitmaps = CPUBitmapGroup(bitmapData, bitmapDims);
    // Acquire GPUBitmap ptrs from CPU Bitmap
    std::vector<const GPUBitmap*> hGPUBitmapPtrs;
    hGPUBitmapPtrs.reserve(bitMapIndex.size());
    for(uint32_t index : bitMapIndex)
    {
        if(index == std::numeric_limits<uint32_t>::max())
            hGPUBitmapPtrs.push_back(nullptr);
        else
            hGPUBitmapPtrs.push_back(bitmaps.Bitmap(index));
    }

    // All loaded to CPU, copy to GPU
    // Alloc Memory
    Vector3f*         dPositions;
    Vector2f*         dUVs;
    QuatF*            dQuats;
    uint64_t*         dIndices;
    bool*             dCulls;
    const GPUBitmap** dAlphaMapPtrs;
    uint64_t*         dOffsets;
    GPUMemFuncs::AllocateMultiData(std::tie(dPositions, dUVs, dQuats,
                                            dIndices, dCulls, dAlphaMapPtrs,
                                            dOffsets),
                                   memory,
                                   {totalDataCount, totalDataCount, totalDataCount,
                                   totalIndexCount, batchRanges.size(), batchRanges.size(),
                                   batchOffsets.size()});
    // Copy Vertex Data
    CUDA_CHECK(cudaMemcpy(dPositions, postitionsCPU.data(),
                          sizeof(Vector3f) * totalDataCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dUVs, uvsCPU.data(),
                          sizeof(Vector2f) * totalDataCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQuats, rotationsCPU.data(),
                          sizeof(QuatF) * totalDataCount,
                          cudaMemcpyHostToDevice));
    // Copy Indices
    CUDA_CHECK(cudaMemcpy(dIndices, indexCPU.data(),
                          totalIndexCount * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    // Cull Face Flags, AlphaMaps & Prim Offsets
    static_assert(sizeof(bool) == sizeof(Byte));
    CUDA_CHECK(cudaMemcpy(dCulls, cullFaceFlags.data(),
                          sizeof(bool) * cullFaceFlags.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dOffsets, batchOffsets.data(),
                          sizeof(uint64_t) * batchOffsets.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAlphaMapPtrs, hGPUBitmapPtrs.data(),
                          sizeof(GPUBitmap*) * hGPUBitmapPtrs.size(),
                          cudaMemcpyHostToDevice));

    // Set Main Pointers of batch
    dData.positions = dPositions;
    dData.uvs = dUVs;
    dData.tbnRotations = dQuats;
    dData.indexList = dIndices;
    dData.cullFace = dCulls;
    dData.alphaMaps = dAlphaMapPtrs;
    dData.primOffsets = dOffsets;
    dData.primBatchCount = static_cast<uint32_t>(batchOffsets.size());
    return e;
}

SceneError GPUPrimitiveTriangle::ChangeTime(const NodeListing&, double,
                                            const SurfaceLoaderGeneratorI&,
                                            const std::string&)
{
    // TODO: Implement
    return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;
}

Vector2ul GPUPrimitiveTriangle::PrimitiveBatchRange(uint32_t surfaceDataId) const
{
    return batchRanges.at(surfaceDataId);
}

AABB3 GPUPrimitiveTriangle::PrimitiveBatchAABB(uint32_t surfaceDataId) const
{
    return batchAABBs.at(surfaceDataId);
}

bool GPUPrimitiveTriangle::PrimitiveBatchHasAlphaMap(uint32_t surfaceDataId) const
{
    return batchAlphaMapFlag.at(surfaceDataId);
}

bool GPUPrimitiveTriangle::PrimitiveBatchBackFaceCulled(uint32_t surfaceDataId) const
{
    return batchBackFaceCullFlag.at(surfaceDataId);
}

uint64_t GPUPrimitiveTriangle::TotalPrimitiveCount() const
{
    return totalPrimitiveCount;
}

uint64_t GPUPrimitiveTriangle::TotalDataCount() const
{
    return totalDataCount;
}

bool GPUPrimitiveTriangle::CanGenerateData(const std::string& s) const
{
    return (s == PrimitiveDataTypeToString(PrimitiveDataType::POSITION) ||
            s == PrimitiveDataTypeToString(PrimitiveDataType::NORMAL) ||
            s == PrimitiveDataTypeToString(PrimitiveDataType::UV));
}

bool GPUPrimitiveTriangle::IsTriangle() const
{
    return true;
}