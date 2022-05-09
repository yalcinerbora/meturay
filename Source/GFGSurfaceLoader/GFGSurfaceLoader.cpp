#include "GFGSurfaceLoader.h"

#include "RayLib/SceneError.h"
#include "RayLib/PrimitiveDataTypes.h"
#include "RayLib/FileSystemUtility.h"

#include <gfg/GFGFileLoader.h>

void GFGSurfaceLoader::LogError(const std::string&)
{

}

GFGSurfaceLoader::GFGSurfaceLoader(const std::string& scenePath,
                                   const std::string& fileExt,
                                   const std::string_view loggerName,
                                   const SceneNodeI& node,
                                   double time)
    : SurfaceLoader(node, time)
    , extension(fileExt)
    , filePath(Utility::MergeFileFolder(scenePath, node.Name()))
    , loggerName(loggerName)
    , file(filePath, std::ios::binary)
    , gfgFileReaderSTL(nullptr)
    , gfgFile(nullptr)
    , innerIds(node.AccessUIntRanged(InnerIdJSON))
{
    if(!file.is_open())
    {
        SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                       "GFG_loader: Unable to open gfg file " + filePath);
    }

    gfgFileReaderSTL = std::make_unique<GFGFileReaderSTL>(file);
    gfgFile = std::make_unique<GFGFileLoader>(gfgFileReaderSTL.get());

    GFGFileError err = gfgFile->ValidateAndOpen();
    if(err != GFGFileError::OK)
    {
        SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                       "GFG_loader: GFG file is corrupted");
    }

    // Check the component logic of the all meshes
    const GFGHeader& h = gfgFile->Header();
    for(unsigned int innerId : innerIds)
    {
        if(innerId >= h.meshes.size())
            throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                 "GFG Loader: Inner index out of range");

        const GFGMeshHeader& mesh = h.meshes[innerId];
        const auto& meshHeader = h.meshes[innerId].headerCore;

        if(meshHeader.topology != GFGTopology::TRIANGLE)
            throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                 "GFG Loader: Only triangle as a primitive is supported");

        // Check components one by one
        bool hasNormals = false;
        bool hasUVs = false;
        bool hasPos = false;
        for(const GFGVertexComponent vc : mesh.components)
        {
            if(vc.internalOffset != 0)
                throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                     "GFG Loader: Vertex data types must be \"struct of arrays\" format");

            switch(vc.logic)
            {
                case GFGVertexComponentLogic::POSITION:
                {
                    hasPos = true;
                    if(vc.dataType != GFGDataType::FLOAT_3)
                        throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                             "GFG Loader: Position data type must be float3");
                    break;
                }
                case GFGVertexComponentLogic::UV:
                {
                    if(hasUVs == true)
                    {
                        METU_LOG("GFG Warning: multiple uvs detected, using the first uv");
                        break;
                    }

                    hasUVs = true;
                    if(vc.dataType != GFGDataType::FLOAT_2)
                        throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                             "GFG Loader: UV data type must be float2");
                    break;
                }
                case GFGVertexComponentLogic::NORMAL:
                {
                    hasNormals = true;
                    if(vc.dataType != GFGDataType::FLOAT_3)
                        throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                             "GFG Loader: Normal data type must be float3");
                    break;
                }
                case GFGVertexComponentLogic::TANGENT:
                {
                    if(vc.dataType != GFGDataType::FLOAT_3)
                        throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                             "GFG Loader: Tangent data type must be float3");
                    break;
                }

                if(!hasNormals)
                    throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                         "Assimp_loader: File does not have normals");
                if(!hasUVs)
                    throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                         "Assimp_loader: File does not have uvs");
                if(!hasPos)
                    throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                         "Assimp_loader: File does not have positions");
            }
        }
    }
}

const char* GFGSurfaceLoader::SufaceDataFileExt() const
{
    return extension.c_str();
}

SceneError GFGSurfaceLoader::AABB(std::vector<AABB3>& list) const
{
    for(unsigned int innerId : innerIds)
    {
        const auto& aabb = gfgFile->Header().meshes[innerId].headerCore.aabb;
        list.push_back(AABB3(Vector3(aabb.min), Vector3(aabb.max)));
    }
    return SceneError::OK;
}

SceneError GFGSurfaceLoader::PrimitiveRanges(std::vector<Vector2ul>& result) const
{
    size_t prevOffset = 0;
    for(unsigned int innerId : innerIds)
    {
        result.emplace_back(prevOffset, 0);
        size_t indexCount = gfgFile->Header().meshes[innerId].headerCore.indexCount;

        // Index count must be multiple of Tri vertex count (3)
        if(indexCount % 3 != 0) return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;
        size_t primitiveCount = indexCount / 3;

        result.back()[1] = result.back()[0] + primitiveCount;
        prevOffset += primitiveCount;
    }
    return SceneError::OK;
}

SceneError GFGSurfaceLoader::PrimitiveCounts(std::vector<size_t>& result) const
{
    for(unsigned int innerId : innerIds)
    {
        size_t indexCount = gfgFile->Header().meshes[innerId].headerCore.indexCount;

        // Index count must be multiple of Tri vertex count (3)
        if(indexCount % 3 != 0) return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;
        size_t primitiveCount = indexCount / 3;

        result.emplace_back(primitiveCount);
    }
    return SceneError::OK;
}

SceneError GFGSurfaceLoader::PrimitiveDataRanges(std::vector<Vector2ul>& result) const
{
    // Self contain indices
    size_t prevOffset = 0;
    for(unsigned int innerId : innerIds)
    {
        result.emplace_back(prevOffset, 0);
        size_t vertexCount = gfgFile->Header().meshes[innerId].headerCore.vertexCount;
        result.back()[1] = result.back()[0] + vertexCount;
        prevOffset += vertexCount;
    }
    return SceneError::OK;
}

SceneError GFGSurfaceLoader::GetPrimitiveData(Byte* result, PrimitiveDataType primitiveDataType) const
{
    // Self contain indices
    uint64_t offset = 0;
    Byte* meshStart = result;

    // GFG API has get all mesh data etc
    // (however we could not use it) since we may not use all of the meshes
    // inside a GFG file
    for(unsigned int innerId : innerIds)
    {
        const auto& mHeader = gfgFile->Header().meshes[innerId];
        const auto& hCore = mHeader.headerCore;;

        switch(primitiveDataType)
        {
            case PrimitiveDataType::POSITION:
            {
                // We've checked that the data is compatible
                // we can just copy here
                gfgFile->MeshVertexComponentDataGroup(result, innerId,
                                                      GFGVertexComponentLogic::POSITION);

                size_t dataSize = gfgFile->MeshVertexComponentDataGroupSize(innerId, GFGVertexComponentLogic::POSITION);

                if(dataSize != (sizeof(Vector3) * hCore.vertexCount))
                    return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

                meshStart += dataSize;
                break;
            }
            case PrimitiveDataType::NORMAL:
            {
                // We've checked that the data is compatible
                // we can just copy here
                gfgFile->MeshVertexComponentDataGroup(result, innerId,
                                                      GFGVertexComponentLogic::NORMAL);

                size_t dataSize = gfgFile->MeshVertexComponentDataGroupSize(innerId, GFGVertexComponentLogic::NORMAL);

                if(dataSize != (sizeof(Vector3) * hCore.vertexCount))
                    return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

                meshStart += dataSize;
                break;
            }
            case PrimitiveDataType::TANGENT:
            {
                // We've checked that the data is compatible
                // we can just copy here
                gfgFile->MeshVertexComponentDataGroup(result, innerId,
                                                      GFGVertexComponentLogic::TANGENT);

                size_t dataSize = gfgFile->MeshVertexComponentDataGroupSize(innerId, GFGVertexComponentLogic::TANGENT);

                if(dataSize != (sizeof(Vector3) * hCore.vertexCount))
                    return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

                meshStart += dataSize;
                break;
            }
            case PrimitiveDataType::BITANGENT:
            {
                // We've checked that the data is compatible
                // we can just copy here
                gfgFile->MeshVertexComponentDataGroup(result, innerId,
                                                      GFGVertexComponentLogic::BINORMAL);

                size_t dataSize = gfgFile->MeshVertexComponentDataGroupSize(innerId, GFGVertexComponentLogic::BINORMAL);

                if(dataSize != (sizeof(Vector3) * hCore.vertexCount))
                    return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

                meshStart += dataSize;
                break;
            }
            case PrimitiveDataType::UV:
            {
                // We've checked that the data is compatible
                // we can just copy here
                gfgFile->MeshVertexComponentDataGroup(result, innerId,
                                                      GFGVertexComponentLogic::UV);

                size_t dataSize = gfgFile->MeshVertexComponentDataGroupSize(innerId, GFGVertexComponentLogic::UV);

                if(dataSize != (sizeof(Vector2) * hCore.vertexCount))
                    return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;

                meshStart += dataSize;
                break;
            }
            case PrimitiveDataType::VERTEX_INDEX:
            {
                // METURay has its index data as uint64_t
                // Check if index is uint64 then just memcpy
                if(hCore.indexSize == sizeof(uint64_t))
                {
                    // We can just memcpy here
                    gfgFile->MeshIndexData(result, innerId);
                }
                else
                {
                    // Well we need to convert here
                    auto ConvertFunc = [&]<class T>()
                    {
                        std::vector<T> data(hCore.indexCount);
                        gfgFile->MeshIndexData(reinterpret_cast<Byte*>(data.data()), innerId);
                        // Convert to 64-bit
                        uint64_t* indexStart = reinterpret_cast<uint64_t*>(meshStart);
                        for(uint64_t i = 0; i < hCore.indexCount; i++)
                        {
                            // Convert one by one
                            indexStart[i] = offset + data[i];
                        }
                    };

                    if(hCore.indexSize == sizeof(uint32_t))
                        ConvertFunc.operator()<uint32_t>();
                    else if(hCore.indexSize == sizeof(uint16_t))
                        ConvertFunc.operator()<uint16_t>();
                }
                // Set offsets manually
                offset += hCore.vertexCount;
                meshStart += sizeof(uint64_t) * hCore.indexCount;
                break;
            }
            default:
                return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
        }
    }
    return SceneError::OK;
}

SceneError GFGSurfaceLoader::HasPrimitiveData(bool& r, PrimitiveDataType primitiveDataType) const
{
    r = true;

    // Convert primitive data type to GFG data type
    GFGVertexComponentLogic logic;
    switch(primitiveDataType)
    {
        case PrimitiveDataType::POSITION:
            logic = GFGVertexComponentLogic::POSITION; break;
        case PrimitiveDataType::NORMAL:
            logic = GFGVertexComponentLogic::NORMAL; break;
        case PrimitiveDataType::TANGENT:
            logic = GFGVertexComponentLogic::TANGENT; break;
        case PrimitiveDataType::BITANGENT:
            logic = GFGVertexComponentLogic::BINORMAL; break;
        case PrimitiveDataType::UV:
            logic = GFGVertexComponentLogic::UV; break;
        case PrimitiveDataType::VERTEX_INDEX:
            return SceneError::OK;
        default:
            return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
    }

    // Check all meshes for this component
    for(unsigned int innerId : innerIds)
    {
        const auto& mesh = gfgFile->Header().meshes[innerId];
        bool hasComp = false;
        for(const GFGVertexComponent& vComp : mesh.components)
        {
            if(vComp.logic != logic) continue;
            hasComp = true;
        }
        r &= hasComp;
    }
    return SceneError::OK;
}

SceneError GFGSurfaceLoader::PrimitiveDataCount(size_t& total, PrimitiveDataType primitiveDataType) const
{
    total = 0;

    SceneError e = SceneError::OK;
    bool hasPrimData = false;
    if((e = HasPrimitiveData(hasPrimData, primitiveDataType)) != SceneError::OK)
        return e;

    // Accumulate without checking
    for(unsigned int innerId : innerIds)
    {
        const auto& mesh = gfgFile->Header().meshes[innerId];

        if(primitiveDataType == PrimitiveDataType::VERTEX_INDEX)
            total += mesh.headerCore.indexCount;
        else
            total += mesh.headerCore.vertexCount;
    }
    return SceneError::OK;
}

SceneError GFGSurfaceLoader::PrimDataLayout(PrimitiveDataLayout& result,
                                            PrimitiveDataType primitiveDataType) const
{
    if(primitiveDataType == PrimitiveDataType::POSITION ||
       primitiveDataType == PrimitiveDataType::NORMAL ||
       primitiveDataType == PrimitiveDataType::TANGENT ||
       primitiveDataType == PrimitiveDataType::BITANGENT)
        result = PrimitiveDataLayout::FLOAT_3;
    else if(primitiveDataType == PrimitiveDataType::UV)
        result = PrimitiveDataLayout::FLOAT_2;
    else if(primitiveDataType == PrimitiveDataType::VERTEX_INDEX)
        result = PrimitiveDataLayout::UINT64_1;
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
    return SceneError::OK;
}