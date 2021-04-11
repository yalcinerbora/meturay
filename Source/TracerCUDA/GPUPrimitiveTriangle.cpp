#include "GPUPrimitiveTriangle.h"

#include "RayLib/SceneError.h"
#include "RayLib/Log.h"
#include "RayLib/SurfaceLoaderGenerator.h"
#include "RayLib/SurfaceLoaderI.h"
#include "RayLib/MemoryAlignment.h"

#include "TracerDebug.h"
#include "TextureFunctions.h"

#include <execution>

struct IndexTriplet
{
    uint64_t i[3];

    uint64_t&           operator[](int j) { return i[j]; }
    const uint64_t&     operator[](int j) const { return i[j]; }
};
static_assert(sizeof(IndexTriplet) == sizeof(uint64_t) * 3);

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
    constexpr size_t VertTangentSize = PrimitiveDataLayoutToSize(TANGENT_LAYOUT);
    constexpr size_t VertNormSize = PrimitiveDataLayoutToSize(NORMAL_LAYOUT);
    constexpr size_t IndexSize = PrimitiveDataLayoutToSize(INDEX_LAYOUT);
    constexpr size_t RotationSize = sizeof(QuatF);

    // Stationary buffers
    std::vector<Byte> postitionsCPU(totalDataCount * VertPosSize);
    std::vector<Byte> uvsCPU(totalDataCount * VertUVSize);
    std::vector<Byte> rotationsCPU(totalDataCount * RotationSize);
    std::vector<Byte> indexCPU(totalIndexCount * IndexSize);

    // Temporary buffers (re-allocated per batch)
    std::vector<Vector3> tangents;
    std::vector<Vector3> normals;

    size_t i = 0;
    for(const auto& loader : loaders)
    {
        const SceneNodeI& node = loader->SceneNode();

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
        size_t tangentCount;
        if((e = loader->HasPrimitiveData(hasTangent, PrimitiveDataType::TANGENT)) != SceneError::OK)
            return e;
        if(hasTangent &&
           (e = loader->PrimitiveDataCount(tangentCount, PrimitiveDataType::TANGENT)) != SceneError::OK)
            return e;

        // Access index three by three for data generation
        IndexTriplet* primStart = reinterpret_cast<IndexTriplet*>(indexCPU.data() + offsetIndex * IndexSize);
        IndexTriplet* primEnd = reinterpret_cast<IndexTriplet*>(indexCPU.data() + offsetIndexNext * IndexSize);

        // Manipulate Ptrs of in/out
        QuatF* rotationsOut = reinterpret_cast<QuatF*>(rotationsCPU.data() + offsetVertex * RotationSize);
        Vector3* normalsIn = normals.data();
        if(hasTangent)
        {
            tangents.resize(tangentCount);
            if((e = loader->GetPrimitiveData(reinterpret_cast<Byte*>(tangents.data()),
                                             PrimitiveDataType::TANGENT)) != SceneError::OK)
                return e;

            Vector3* tangentsIn = tangents.data();

            // Utilize tangent and normal for quat generation
            std::for_each(std::execution::par_unseq,
                          primStart, primEnd,
                          [&](IndexTriplet& indices)
                          {
                              Vector3 normals[3];
                              normals[0] = normalsIn[indices[0]];
                              normals[1] = normalsIn[indices[1]];
normals[2] = normalsIn[indices[2]];

Vector3 tangents[3];
tangents[0] = tangentsIn[indices[0]];
tangents[1] = tangentsIn[indices[1]];
tangents[2] = tangentsIn[indices[2]];

//Vector3 biTangents[3];
//biTangents[0] = biTangentsIn[indices[0]];
//biTangents[1] = biTangentsIn[indices[1]];
//biTangents[2] = biTangentsIn[indices[2]];

// Generate rotations
QuatF q0, q1, q2;
Triangle::LocalRotation(q0, q1, q2, normals, tangents);

rotationsOut[indices[0]] = q0;
rotationsOut[indices[1]] = q1;
rotationsOut[indices[2]] = q2;

// Finally accumulate offset for combined vertex buffer usage
if(i != 0)
{
    indices[0] += offsetVertex;
    indices[1] += offsetVertex;
    indices[2] += offsetVertex;
}
                          });
        }
        else
        {
        Vector3* positionsIn = reinterpret_cast<Vector3*>(postitionsCPU.data() + offsetVertex * VertPosSize);
        Vector2* uvsIn = reinterpret_cast<Vector2*>(uvsCPU.data() + offsetVertex * VertUVSize);

        // Utilize position and uv for quat generation
        std::for_each(std::execution::par_unseq,
                      primStart, primEnd,
                      [&](IndexTriplet& indices)
                      {
                          Vector3 normals[3];
                          normals[0] = normalsIn[indices[0]];
                          normals[1] = normalsIn[indices[1]];
                          normals[2] = normalsIn[indices[2]];

                          Vector3 positions[3];
                          positions[0] = positionsIn[indices[0]];
                          positions[1] = positionsIn[indices[1]];
                          positions[2] = positionsIn[indices[2]];

                          Vector2 uvs[3];
                          uvs[0] = uvsIn[indices[0]];
                          uvs[1] = uvsIn[indices[1]];
                          uvs[2] = uvsIn[indices[2]];

                          // Generate rotations
                          QuatF q0, q1, q2;
                          Triangle::LocalRotation(q0, q1, q2, positions, normals, uvs);

                          rotationsOut[indices[0]] = q0;
                          rotationsOut[indices[1]] = q1;
                          rotationsOut[indices[2]] = q2;

                          // Finally accumulate offset for combined vertex buffer usage
                          if(i != 0)
                          {
                              indices[0] += offsetVertex;
                              indices[1] += offsetVertex;
                              indices[2] += offsetVertex;
                          }
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

        if(access != TextureAccessLayout::R ||
           access != TextureAccessLayout::G ||
           access != TextureAccessLayout::B ||
           access != TextureAccessLayout::A)
            return SceneError::BITMAP_LOAD_CALLED_WITH_MULTIPLE_CHANNELS;

        TextureChannelType channel = TextureFunctions::TextureAccessLayoutToTextureChannels(access)[0];

        // Check if this texture/channel pair is already loaded
        auto loc = loadedBitmaps.cend();
        if((loc = loadedBitmaps.find(std::make_pair(textureId, channel))) == loadedBitmaps.cend())
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
    bitmaps = std::move(CPUBitmapGroup(bitmapData, bitmapDims));
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
    size_t posSize = sizeof(Vector3f) * totalDataCount;
    posSize = Memory::AlignSize(posSize);
    size_t uvSize = sizeof(Vector2f) * totalDataCount;
    uvSize = Memory::AlignSize(uvSize);
    size_t quatSize = sizeof(QuatF) * totalDataCount;
    quatSize = Memory::AlignSize(quatSize);
    size_t indexSize = sizeof(uint64_t) * totalIndexCount;
    indexSize = Memory::AlignSize(indexSize);
    size_t cullSize = sizeof(bool) * batchRanges.size();
    cullSize = Memory::AlignSize(cullSize);
    size_t alphaMapPtrSize = sizeof(GPUBitmap*) * batchRanges.size();
    alphaMapPtrSize = Memory::AlignSize(alphaMapPtrSize);
    size_t offsetSize = sizeof(uint64_t) * batchOffsets.size();
    size_t totalSize = (posSize + uvSize +
                        quatSize + indexSize +
                        cullSize + alphaMapPtrSize + 
                        offsetSize);

    memory = std::move(DeviceMemory(totalSize));
    size_t offset = 0;
    Byte* dPositions = static_cast<Byte*>(memory) + offset;
    offset += posSize;
    Byte* dUVs = static_cast<Byte*>(memory) + offset;
    offset += uvSize;
    Byte* dQuats = static_cast<Byte*>(memory) + offset;
    offset += quatSize;
    Byte* dIndices = static_cast<Byte*>(memory) + offset;
    offset += indexSize;    
    Byte* dCulls = static_cast<Byte*>(memory) + offset;
    offset += cullSize;
    Byte* dAlphaMapPtrs = static_cast<Byte*>(memory) + offset;
    offset += alphaMapPtrSize;
    Byte* dOffsets = static_cast<Byte*>(memory) + offset;
    offset += offsetSize;
    assert(offset == totalSize);

    // Copy Vertex Data
    CUDA_CHECK(cudaMemcpy(dPositions,  postitionsCPU.data(),
                          sizeof(Vector3f)  * totalDataCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dUVs, uvsCPU.data(),
                          sizeof(Vector2f)* totalDataCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQuats, rotationsCPU.data(),
                          sizeof(QuatF)* totalDataCount,
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
    dData.positions = reinterpret_cast<Vector3f*>(dPositions);
    dData.uvs = reinterpret_cast<Vector2f*>(dUVs);
    dData.tbnRotations = reinterpret_cast<QuatF*>(dQuats);
    dData.indexList = reinterpret_cast<uint64_t*>(dIndices);
    dData.cullFace = reinterpret_cast<bool*>(dCulls);
    dData.alphaMaps = reinterpret_cast<const GPUBitmap**>(dAlphaMapPtrs);
    dData.primOffsets = reinterpret_cast<uint64_t*>(dOffsets);
    dData.primBatchCount = static_cast<uint32_t>(batchOffsets.size());

    return e;
}

SceneError GPUPrimitiveTriangle::ChangeTime(const NodeListing& surfaceDataNodes, double time,
                                            const SurfaceLoaderGeneratorI& loaderGen,
                                            const std::string& scenePath)
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

PrimTransformType GPUPrimitiveTriangle::TransformType() const
{
    return PrimTransformType::CONSTANT_LOCAL_TRANSFORM;
}

bool GPUPrimitiveTriangle::CanGenerateData(const std::string& s) const
{
    return (s == PrimitiveDataTypeToString(PrimitiveDataType::POSITION) ||
            s == PrimitiveDataTypeToString(PrimitiveDataType::NORMAL) ||
            s == PrimitiveDataTypeToString(PrimitiveDataType::UV));
}