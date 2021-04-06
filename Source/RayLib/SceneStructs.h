#pragma once

#include <vector>
#include <array>
#include <set>
#include <map>

#include "Vector.h"
#include "Matrix.h"
#include "SceneError.h"
#include "Types.h"
#include "HitStructs.h"

//enum class FilterType
//{
//    LINEAR,
//    NEAREST,
//
//    END
//};
//
//enum class TextureType
//{
//    TEX_1D,
//    TEX_2D,
//    TEX_3D,
//    CUBE,
//
//    END
//};

using NodeId = uint32_t;        // Node Id is generic name of the id logic
using MaterialId = uint32_t;    // Material Id represent material of some kind
using SurfaceDataId = uint32_t; // Surface Id represent up to "MaxPrimitivePerSurface"
                                // material-primitive pairings of some kind

using NodeIndex = uint32_t;     // Node index is the index of the node that is on the list
                                // This is unique only for each type. (Materials, Primitives etc.)
using InnerIndex = uint32_t;    // Inner Index is sub index of the node
                                // Each node can define multiple ids

using TypeIdPair = std::pair<std::string, uint32_t>;
using IdPair = std::pair<uint32_t, uint32_t>;
using IdPairs = std::array<IdPair, SceneConstants::MaxPrimitivePerSurface>;
using IdList = std::vector<uint32_t>;

using IdPairsWithAnId = std::pair<uint32_t, IdPairs>;

class SceneNodeI;

enum class TextureAccessLayout
{
    R, G, B, A,
    RG,
    RGB,
    RGBA
    // TODO: add more here for swizzle access etc.
};

// Compiled Data which will be used to create actual class later
struct AccelGroupData
{
    std::string                     accelType;
    std::string                     primType;
    std::map<uint32_t, IdPairs>     matPrimIdPairs;
    std::vector<uint32_t>           transformIds;
    std::unique_ptr<SceneNodeI>     accelNode;
};

struct WorkBatchData
{
    std::string                     primType;
    std::string                     matType;
    std::set<MaterialId>            matIds;
};

// Construction data is used to create camera or lights
// SceneNode Interface is used singular in this case
// meaning only single element on the node is enabled
struct CameraConstructionData
{
    uint32_t                        transformId;
    uint32_t                        mediumId;
    uint32_t                        constructionId;
    std::unique_ptr<SceneNodeI>     node;
};

struct LightConstructionData : public CameraConstructionData
{
    uint32_t                        materialId;
};

struct LightGroupData
{
    bool                                isPrimitive;
    std::string                         primTypeName;
    std::vector<LightConstructionData>  constructionInfo;
};

using LightGroupDataList = std::vector<LightConstructionData>;
using CameraGroupDataList = std::vector<CameraConstructionData>;

using MaterialKeyListing = std::map<TypeIdPair, HitKey>;

struct EndpointStruct
{
    uint32_t        acceleratorId;
    uint32_t        transformId;
    uint32_t        materialId;
    uint32_t        mediumId;
};

struct SurfaceStruct
{
    static constexpr int MATERIAL_INDEX = 0;
    static constexpr int PRIM_INDEX = 1;

    uint32_t        acceleratorId;
    uint32_t        transformId;
    IdPairs         matPrimPairs;
    int8_t          pairCount;
};

struct LightSurfaceStruct
{
    bool        isPrimitive;
    uint32_t    mediumId;
    uint32_t    transformId;
    uint32_t    acceleratorId;
    uint32_t    materialId;
    uint32_t    lightOrPrimId;
};

struct CameraSurfaceStruct
{
    uint32_t    mediumId;
    uint32_t    transformId;
    uint32_t    cameraId;
};

struct NodeTextureStruct
{
    uint32_t            texId;
    TextureAccessLayout channelLayout;
};

struct TextureStruct
{
    uint32_t    texId;
    std::string filePath;
};