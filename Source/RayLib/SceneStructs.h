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

class GPUDistribution2D;

//enum class LightType : uint16_t
//{
//    POINT,
//    DIRECTIONAL,
//    SPOT,
//    RECTANGULAR,
//    TRIANGULAR,
//    DISK,
//    SPHERICAL,
//    PRIMITIVE,
//    //
//    END
//
//};
//
//enum class TransformType : uint8_t
//{
//    MATRIX,         // 4x4 Transformation matrix
//    TRS,            // Transform Rotate Scale
//    ROTATION,       // Rotation only
//    TRANSLATION,    // Translation only
//};

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

struct LightGroupData
{
    std::unique_ptr<SceneNodeI>     lightNode;

    bool                            isPrimitive;
    uint32_t                        transformId;
    uint32_t                        mediumId;
    uint32_t                        lightOrPrimitiveId;
    
};

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