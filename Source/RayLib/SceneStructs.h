#pragma once

#include <vector>
#include <array>
#include <set>
#include <map>
#include <tuple>

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

using IdKeyPair = std::pair<uint32_t, HitKey>;
using IdKeyPairs = std::array<IdKeyPair, SceneConstants::MaxPrimitivePerSurface>;

using TypeIdIdTriplet = std::tuple<std::string, uint32_t, uint32_t>;

//using IdList = std::vector<uint32_t>;
//
//using IdPairsWithAnId = std::pair<uint32_t, IdPairs>;

using IndexLookup = std::map<NodeId, std::pair<NodeIndex, InnerIndex>>;

class SceneNodeI;

enum class TextureChannelType
{
    R,
    G,
    B,
    A
};

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
    struct SurfaceDef
    {
        uint32_t    transformId;
        IdPairs     matPrimIdPairs;
    };
    struct LSurfaceDef
    {
        uint32_t    transformId;
        uint32_t    primId;
        uint32_t    lightId;
    };

    std::string                     accelType;
    std::string                     primType;
    // List of Surfaces
    std::map<uint32_t, SurfaceDef>  surfaces;
    // If available List of light surfaces
    std::map<uint32_t, LSurfaceDef> lightSurfaces;
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
struct EndpointConstructionData
{
    uint32_t                        surfaceId;
    uint32_t                        transformId;
    uint32_t                        mediumId;
    uint32_t                        endpointId;
    uint32_t                        primitiveId;
    std::unique_ptr<SceneNodeI>     node;
};

struct LightGroupData
{
    std::string                             primTypeName;
    std::vector<EndpointConstructionData>   constructionInfo;

    bool IsPrimitiveLight() const { return primTypeName.empty(); }
};

using EndpointGroupDataList = std::vector<EndpointConstructionData>;

using MaterialKeyListing = std::map<TypeIdPair, HitKey>;
using BoundaryMaterialKeyListing = std::map<TypeIdIdTriplet, HitKey>;

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
    uint32_t    primId;
    uint32_t    lightId;
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
    bool        isSigned;
    std::string filePath;
};

using TextureNodeMap = std::map<uint32_t, TextureStruct>;