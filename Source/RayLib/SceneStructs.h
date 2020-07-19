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

enum class LightType
{
    POINT,
    DIRECTIONAL,
    SPOT,
    RECTANGULAR,
    TRIANGULAR,
    DISK,
    SPHERICAL,
    PRIMITIVE,
    //
    END

};

enum class FilterType
{
    LINEAR,
    NEAREST,

    END
};

enum class TextureType
{
    TEX_1D,
    TEX_2D,
    TEX_3D,
    CUBE,

    END
};

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

class SceneNodeI;

// Compiled Data which will be used to create actual class later
struct AccelGroupData
{
    std::string                     accelType;
    std::string                     primType;
    std::map<uint32_t, IdPairs>     matPrimIdPairs;
    std::unique_ptr<SceneNodeI>     accelNode;
};

struct WorkBatchData
{
    std::string             primType;
    std::string             matType;
    std::set<MaterialId>    matIds;
};

using MaterialKeyListing = std::map<TypeIdPair, HitKey>;

// CPU Representation of Light
// No inheritance here
struct CPULight
{
    LightType       type;
    HitKey          matKey;
    PrimitiveId     primId;
    Vector3         flux;
    Vector3         position0;
    Vector3         position1;
    Vector3         position2;
};

struct CPUMedium
{    
    Vector3 sigmaA, sigmaS;     // Scattering and Absorbtion Coeffs
    float phase;
    float index;                // IoR
};

using GPUTransform = Matrix4x4;

struct SurfaceStruct
{
    uint32_t        acceleratorId;
    uint32_t        transformId;
    IdPairs         matPrimPairs;
    int8_t          pairCount;
};