#include "GDebugRendererSVO.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <execution>
#include <atomic>
#include <Imgui/imgui.h>

#include "RayLib/FileSystemUtility.h"
#include "RayLib/Log.h"
#include "RayLib/RandomColor.h"
#include "RayLib/CoordinateConversion.h"
#include "RayLib/BitManipulation.h"

#include "TextureGL.h"
#include "GuideDebugStructs.h"
#include "GuideDebugGUIFuncs.h"
#include "GLConversionFunctions.h"

uint64_t ExpandTo64(uint32_t val)
{
    // https://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit
    uint64_t x = val;
    x &= 0x1fffff;
    x = (x | x << 32) & 0x001f00000000ffff;
    x = (x | x << 16) & 0x001f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

uint64_t ComposeMortonCode64(const Vector3ui& val)
{
    uint64_t x = ExpandTo64(val[0]);
    uint64_t y = ExpandTo64(val[1]);
    uint64_t z = ExpandTo64(val[2]);
    return ((x << 0) | (y << 1) | (z << 2));
}

bool SVOctree::IsChildrenLeaf(uint64_t packedData)
{
    return static_cast<bool>((packedData >> IS_LEAF_OFFSET) & IS_LEAF_BIT_COUNT);
}

uint32_t SVOctree::ChildMask(uint64_t packedData)
{
    return static_cast<uint32_t>((packedData >> CHILD_MASK_OFFSET) & CHILD_MASK_BIT_MASK);
}

uint32_t SVOctree::ChildrenCount(uint64_t packedData)
{
    return Utility::BitCount(ChildMask(packedData));
}

uint32_t SVOctree::ChildrenIndex(uint64_t packedData)
{
    return static_cast<uint32_t>((packedData >> CHILD_OFFSET) & CHILD_BIT_MASK);
}

uint32_t SVOctree::ParentIndex(uint64_t packedData)
{
    return static_cast<uint32_t>((packedData >> PARENT_OFFSET) & PARENT_BIT_MASK);
}

uint32_t SVOctree::FindChildOffset(uint64_t packedData, uint32_t childId)
{
    assert(childId < 8);
    uint32_t mask = ChildMask(packedData);
    uint32_t bitLoc = (1 << static_cast<uint32_t>(childId));
    assert((bitLoc & mask) != 0);
    uint32_t lsbMask = bitLoc - 1;
    uint32_t offset = Utility::BitCount(lsbMask & mask);
    assert(offset < 8);
    return offset;
}

bool SVOctree::HasChild(uint64_t packedData, uint32_t childId)
{
    assert(childId < 8);
    uint32_t childMask = ChildMask(packedData);
    return (childMask >> childId) & 0b1;
}

//
//Vector3f SVOctree::VoxelDirection(uint32_t directionId)
//{
//    static constexpr Vector3f X_AXIS = XAxis;
//    static constexpr Vector3f Y_AXIS = YAxis;
//    static constexpr Vector3f Z_AXIS = ZAxis;
//
//    int8_t signX = (directionId >> 0) & 0b1;
//    int8_t signY = (directionId >> 1) & 0b1;
//    int8_t signZ = (directionId >> 2) & 0b1;
//    signX = (1 - signX) * 2 - 1;
//    signY = (1 - signY) * 2 - 1;
//    signZ = (1 - signZ) * 2 - 1;
//
//    Vector3f dir = (X_AXIS * signX +
//                    Y_AXIS * signY +
//                    Z_AXIS * signZ);
//    return dir.Normalize();
//}
//
Vector4uc SVOctree::DirectionToAnisoLocations(Vector2f& interp,
                                              const Vector3f& direction)
{
    // I couldn't comprehend this as a mathematical
    // representation so tabulated the output
    static constexpr Vector4uc TABULATED_LAYOUTS[12] =
    {
        Vector4uc(0,1,0,1), Vector4uc(1,2,1,2),  Vector4uc(2,3,2,3), Vector4uc(3,0,3,0),
        Vector4uc(0,1,4,5), Vector4uc(1,2,5,6),  Vector4uc(2,3,6,7), Vector4uc(3,0,7,4),
        Vector4uc(4,5,4,5), Vector4uc(5,6,5,6),  Vector4uc(6,7,6,7), Vector4uc(7,4,7,4)
    };

    static constexpr float PIXEL_X = 4;
    static constexpr float PIXEL_Y = 2;

    Vector3 dirZUp = Vector3(direction[2], direction[0], direction[1]);
    Vector2f thetaPhi = Utility::CartesianToSphericalUnit(dirZUp);
    // Normalize to generate UV [0, 1]
    // theta range [-pi, pi]
    float u = (thetaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
    // phi range [0, pi]
    float v = 1.0f - (thetaPhi[1] / MathConstants::Pi);

    // Convert to pixelCoords
    float pixelX = u * PIXEL_X;
    float pixelY = v * PIXEL_Y;

    float indexX;
    float interpX = modff(pixelX, &indexX);
    uint32_t indexXInt = (indexX >= 4) ? 0 : static_cast<uint32_t>(indexX);

    float indexY;
    float interpY = abs(modff(pixelY + 0.5f, &indexY));
    uint32_t indexYInt = static_cast<uint32_t>(indexY);

    interp = Vector2f(interpX, interpY);
    return TABULATED_LAYOUTS[indexYInt * 4 + indexXInt];
}

float SVOctree::AnisoRadianceF::Read(uint8_t index) const
{
    uint8_t iMSB = index >> 2;
    uint8_t iLower = index & 0b11;
    return data[iMSB][iLower];
}

float SVOctree::AnisoRadianceF::Read(const Vector4uc& indices,
                                     const Vector2f& interp) const
{
    // Bilinear interpolation
    float a = HybridFuncs::Lerp(Read(indices[0]), Read(indices[1]), interp[0]);
    float b = HybridFuncs::Lerp(Read(indices[2]), Read(indices[3]), interp[0]);
    float result = HybridFuncs::Lerp(a, b, interp[1]);
    return result;
}

float SVOctree::TraceRay(uint32_t& leafId, const RayF& ray,
                         float tMin, float tMax) const
{
    static constexpr float EPSILON = MathConstants::Epsilon;
    // We wrap the exp2f function here
    // since you can implement an integer version
    // with some bit manipulation
    // It is maybe faster?
    auto FastNegExp2f = [](int power)
    {
        //// Just manipulate exponent bits
        //static constexpr int IEEE754_EXPONENT_OFFSET = 23;
        //static constexpr int IEEE754_EXPONENT_BIAS = 127;
        //// This expression probably make it slow tho
        //// 2^0 does not fit the mathematical expression
        //// We need a special case here
        //if(power == 0) return 1.0f;
        //uint32_t fVal = IEEE754_EXPONENT_BIAS - power;
        //fVal <<= IEEE754_EXPONENT_OFFSET;
        //return __int_as_float(fVal);

        // This is the float version
        return std::log2f(-static_cast<float>(power));
    };
    // Instead of holding a large stack for each thread,
    // we hold a bit stack which will hold the node morton Id
    // when we "pop" from the stack we will use global parent pointer
    // to pop up a level then use the extracted bits (last 3 bits)
    // to find the voxel corner position
    //
    // An example
    // Root corner is implicitly (0, 0) (2D example)
    // If morton stack has 11
    // corner = 0 + exp2f(-(morton3BitCount + 1)) if(axis bit is 1 else 0)
    // for each dimension
    // if we pop up a level from (0.5, 0.5) and the bit are 11
    // p = 0.5 - exp2f(-(morton3BitCount + 1))  if(axis bit is 1 else 0)
    uint64_t mortonBitStack = 0;
    uint8_t stack3BitCount = 0;
    // Helper Lambdas for readability (Basic stack functionality over the variables)
    auto PushMortonCode = [&mortonBitStack, &stack3BitCount](uint32_t bitCode)
    {
        assert(bitCode < 8);
        mortonBitStack = (mortonBitStack << 3) | bitCode;
        stack3BitCount += 1;
        assert(stack3BitCount <= 21);
    };
    auto PopMortonCode = [&mortonBitStack, &stack3BitCount]()
    {
        mortonBitStack >>= 3;
        stack3BitCount -= 1;
    };
    auto ReadMortonCode = [&mortonBitStack]() -> uint32_t
    {
        return mortonBitStack & 0b111;
    };

    // Set leaf to invalid value
    leafId = UINT32_MAX;

    // Pull ray to the registers
    Vector3f rDir = ray.getDirection();
    Vector3f rPos = ray.getPosition();
    // On AABB intersection tests (Voxels are special case AABBs)
    // we will use 1 / rayDirection in their calculations
    // If any of the directions are parallel to any of the axes
    // t values will explode (or NaN may happen (0 / 0))
    // In order to prevent that use an epsilon here
    if(fabs(rDir[0]) < EPSILON) rDir[0] = copysignf(EPSILON, rDir[0]);
    if(fabs(rDir[1]) < EPSILON) rDir[1] = copysignf(EPSILON, rDir[1]);
    if(fabs(rDir[2]) < EPSILON) rDir[2] = copysignf(EPSILON, rDir[2]);

    // Ray is in world space convert to SVO space
    // (SVO is in [0,1])
    // Translate the ray to -AABBmin
    // Scale the ray from  [AABBmin, AABBmax] to [0,1]
    const Vector3f svoTranslate = -svoAABB.Min();
    const Vector3f svoScale = Vector3f(1.0f) / (svoAABB.Max() - svoAABB.Min());
    rDir = rDir * svoScale;
    rPos = (rPos + svoTranslate) * svoScale;
    // Now voxel span calculations become trivial
    // 1.0f, 0.5f, 0.25f ...
    // You can use integer exp2f version now

    // Now also align all the rays as if their direction is positive
    // on all axes and we will
    // mirror the SVO appropriately
    rPos[0] = (rDir[0] < 0.0f) ? (1.0f - rPos[0]) : rPos[0];
    rPos[1] = (rDir[1] < 0.0f) ? (1.0f - rPos[1]) : rPos[1];
    rPos[2] = (rDir[2] < 0.0f) ? (1.0f - rPos[2]) : rPos[2];
    // Mask will flip the childId to mirrored childId
    // XOR the child bit mask with this value when traversing
    uint8_t mirrorMask = 0b000;
    if(rDir[0] < 0.0f) mirrorMask |= 0b001;
    if(rDir[1] < 0.0f) mirrorMask |= 0b010;
    if(rDir[2] < 0.0f) mirrorMask |= 0b100;

    // Generate Direction Coefficient (All ray directions are positive now
    // so abs the direction as well)
    const Vector3f dirCoeff = Vector3f(1.0f) / rDir.Abs();
    // Every AABB(Voxel) Test will have this formula
    //
    // (planes - rayPos) * dirCoeff - tCurrent
    //
    // In order to hit compiler to use mad (multiply and add) instructions,
    // expand the parenthesis and save the "rayPos * dirCoeff"
    //
    // planes * dirCoefff - rayPos - tCurrent = 0
    //
    // planes * dirCoefff - rayPos = tCurrent
    // we can check if this is greater or smaller etc.
    const Vector3f posBias = rPos * dirCoeff;

    // Since we mirrored the svo and rays are always goes through
    // as positive, we can ray march only by checking the "tMin"
    // (aka. "Bottom Left" corner)
    // We will only check tMax to find if we leaved the voxel

    // Do initial AABB check with ray's tMin
    // Only with lower left corner (first iteration will check
    // the upper right corner and may terminate)
    Vector3f tMinXYZ = (-posBias);
    // Since every ray has positive direction, "tMin" will have the
    // minimum values. Maximum of "tMin" can be used to advance the
    // ray.
    // Normally for every orientation (wrt. ray dir)
    // you need to check the top right corner as well
    float tSVO = std::max(std::max(tMinXYZ[0], tMinXYZ[1]), tMinXYZ[2]);
    // Move the point to the "bottom left" corners
    // Only do this if and of the three axes are not negative
    if(!(tMinXYZ < Vector3f(tMin)))
    {
        tMin = tSVO;
        // Nudge the tMin here
        tMin = nextafter(tMin, INFINITY);
    }
    // Traversal initial data
    // 0 index is root
    uint32_t nodeId = 0;
    // Initially corner is zero (SVO space [0,1])
    Vector3f corner = Zero3f;
    // Empty node boolean will be used to
    // create cleaner code
    // There should be at least one node
    // available on the tree (root)
    bool emptyNode = false;
    // Ray March Loop
    while(nodeId != INVALID_PARENT)
    {
        // Children Voxel Span (derived from the level)
        float childVoxSpan = FastNegExp2f(stack3BitCount + 1);
        float currentVoxSpan = FastNegExp2f(stack3BitCount);
        // Find Center
        Vector3f voxCenter = corner + Vector3f(childVoxSpan);
        Vector3f voxTopRight = corner + Vector3f(currentVoxSpan);

        // Check early pop
        Vector3f tMaxXYZ = (voxTopRight * dirCoeff - posBias);
        float tMax = std::min(std::min(tMaxXYZ[0], tMaxXYZ[1]), tMaxXYZ[2]);

        // We will "fake" traverse this empty node by just advancing
        // the tMin
        if(emptyNode)
        {
            // Find the smallest non-negative tValue
            // Advance the tMin accordingly
            tMin = tMax;
            // Nudge the tMin here
            tMin = nextafter(tMin, INFINITY);
            uint32_t mortonBits = ReadMortonCode();
            corner -= Vector3f(((mortonBits >> 0) & 0b1) ? currentVoxSpan : 0.0f,
                               ((mortonBits >> 1) & 0b1) ? currentVoxSpan : 0.0f,
                               ((mortonBits >> 2) & 0b1) ? currentVoxSpan : 0.0f);
            PopMortonCode();
            emptyNode = false;
            continue;
        }

        // We can fetch the node now
        uint64_t node = nodes[nodeId];

        // Actual Traversal code
        // If out of bounds directly pop to parent
        if(tMax < tMin)
        {
            // Update the corner back according to the stack
            uint32_t mortonBits = ReadMortonCode();
            corner -= Vector3f(((mortonBits >> 0) & 0b1) ? currentVoxSpan : 0.0f,
                               ((mortonBits >> 1) & 0b1) ? currentVoxSpan : 0.0f,
                               ((mortonBits >> 2) & 0b1) ? currentVoxSpan : 0.0f);
            PopMortonCode();
            // Actually pop to the parent
            nodeId = ParentIndex(node);
            emptyNode = false;
        }
        // Check the children etc.
        else
        {
            uint8_t childId = 0;
            uint8_t mirChildId = 0;
            // Find the t values of the voxel center
            // Find out the childId using that
            Vector3f tMinXYZ = (voxCenter * dirCoeff - posBias);
            if(tMinXYZ[0] < tMin) childId |= 0b001;
            if(tMinXYZ[1] < tMin) childId |= 0b010;
            if(tMinXYZ[2] < tMin) childId |= 0b100;
            // Check if this childId has actually have a child
            // Don't forget to mirror the SVO
            mirChildId = childId ^ mirrorMask;

            // If the found out voxel has children
            // Traverse down
            if(HasChild(node, mirChildId))
            {
                // Descend down
                // Only update the node if it has an actual children
                nodeId = ChildrenIndex(node) + FindChildOffset(node, mirChildId);

                // If it is leaf just return the "nodeId" it is actually
                // leaf id
                if(IsChildrenLeaf(node))
                {
                    leafId = nodeId;
                    break;
                }
            }
            // If this node does not have a child
            // "Fake" traverse down to child node
            // (don't update the node variable)

            // Set empty node for next iteration
            emptyNode = !HasChild(node, mirChildId);
            // Unfortunately it is not leaf
            // Traverse down, update Corner
            corner += Vector3f(((childId >> 0) & 0b1) ? childVoxSpan : 0.0f,
                               ((childId >> 1) & 0b1) ? childVoxSpan : 0.0f,
                               ((childId >> 2) & 0b1) ? childVoxSpan : 0.0f);
            // Push the non-mirrored child id to the mask
            // (since corners are in mirrored coordinates)
            PushMortonCode(childId);
        }
    }
    // All Done!
    return tMin;
}

bool SVOctree::LeafIndex(uint32_t& index, const Vector3f& worldPos,
                         bool checkNeighbours) const
{
    // Useful constants
    static constexpr uint32_t DIMENSION = 3;
    static constexpr uint32_t DIM_MASK = (1 << DIMENSION) - 1;

    // Descend module (may be called multiple times
    // when checkNeighbours is on)
    auto Descend = [&](uint64_t mortonCode) -> bool
    {
        uint32_t mortonLevelShift = (leafDepth - 1) * DIMENSION;
        // Now descend down
        uint32_t currentNodeIndex = 0;
        for(uint32_t i = 0; i < leafDepth; i++)
        {
            uint64_t currentNode = nodes[currentNodeIndex];
            uint32_t childId = (mortonCode >> mortonLevelShift) & DIM_MASK;
            // Check if this node has that child avail
            if(!((ChildMask(currentNode) >> childId) & 0b1))
                return false;

            uint32_t childOffset = FindChildOffset(currentNode, childId);
            uint32_t childrenIndex = ChildrenIndex(currentNode);
            currentNodeIndex = childrenIndex + childOffset;
            mortonLevelShift -= DIMENSION;
        }
        // If we descended down properly currentNode should point to the
        // index. Notice that, last level's children ptr will point to the leaf arrays
        index = currentNodeIndex;
        return true;
    };

    index = UINT32_MAX;
    if(svoAABB.IsOutside(worldPos))
    {
        index = UINT32_MAX;
        return false;
    }

    // Calculate Dense Voxel Id
    Vector3f lIndex = ((worldPos - svoAABB.Min()) / leafVoxelSize);
    Vector3f lIndexInt;
    Vector3f lIndexFrac = Vector3f(modff(lIndex[0], &(lIndexInt[0])),
                                   modff(lIndex[1], &(lIndexInt[1])),
                                   modff(lIndex[2], &(lIndexInt[2])));
    lIndex.FloorSelf();
    // Find movement mask
    // If that bit is set we go negative instead of positive
    Vector3ui inc = Vector3ui((lIndexFrac[0] < 0.5f) ? -1 : 1,
                              (lIndexFrac[1] < 0.5f) ? -1 : 1,
                              (lIndexFrac[2] < 0.5f) ? -1 : 1);

    // Shuffle the order as well
    // Smallest fraction (w.r.t 0.5f)
    lIndexFrac = Vector3f(0.5f) - (lIndexFrac - Vector3f(0.5f)).Abs();
    // Classic bit traversal (for loop below)
    // has XYZ axis ordering (X checked first etc.)
    // we need to shuffle the order so that the closest axis will be checked
    // first
    int minIndex0 = lIndexFrac.Min();
    int minIndex1 = Vector2f(lIndexFrac[(minIndex0 + 1) % 3],
                             lIndexFrac[(minIndex0 + 2) % 3]).Min();
    minIndex1 = (minIndex0 + minIndex1 + 1) % 3;
    int minIndex2 = 3 - minIndex0 - minIndex1;
    assert((minIndex0 != minIndex1) &&
           (minIndex1 != minIndex2));
    assert(minIndex0 >= 0 && minIndex0 < 3 &&
           minIndex1 >= 0 && minIndex1 < 3 &&
           minIndex2 >= 0 && minIndex2 < 3);

    // Initial Level index
    Vector3ui denseIndex = Vector3ui(lIndex[0], lIndex[1], lIndex[2]);

    bool found = false;
    for(uint32_t i = 0; i < 8; i++)
    {
        // Convert if clauses to mathematical expression
        Vector3ui curIndex = Vector3ui(((i >> 0) & 0b1) ? 1 : 0,
                                       ((i >> 1) & 0b1) ? 1 : 0,
                                       ((i >> 2) & 0b1) ? 1 : 0);
        // Shuffle the iteration values for the traversal
        // Nearest nodes will be checked first
        Vector3ui curShfl;
        curShfl[minIndex0] = curIndex[0];
        curShfl[minIndex1] = curIndex[1];
        curShfl[minIndex2] = curIndex[2];

        Vector3ui voxIndex = Vector3ui(denseIndex[0] + curShfl[0] * inc[0],
                                       denseIndex[1] + curShfl[1] * inc[1],
                                       denseIndex[2] + curShfl[2] * inc[2]);

        // Generate Morton code of the index
        uint64_t voxelMorton = ComposeMortonCode64(voxIndex);
        // Traverse this morton code
        found = Descend(voxelMorton);
        // Terminate if we are only checking a single voxel
        // or a voxel is found
        if(!checkNeighbours || found) break;
    }
    return found;
}

float SVOctree::ReadRadiance(uint32_t nodeId, bool isLeaf,
                             const Vector3f& outgoingDir) const
{
    if(nodeId == UINT32_MAX)
    {
        // TODO: sample the boundary light in this case
        return 0.0f;
    }

    Vector2f interpValues;
    Vector4uc neighbours = DirectionToAnisoLocations(interpValues,
                                                     outgoingDir);

    const auto& gRadRead = (isLeaf) ? leafRadianceRead
                                    : radianceRead;
    return gRadRead[nodeId].Read(neighbours, interpValues);
}

GDebugRendererSVO::GDebugRendererSVO(const nlohmann::json& config,
                                     const TextureGL& gradientTexture,
                                     const std::string& configPath,
                                     uint32_t depthCount)
    : linearSampler(SamplerGLEdgeResolveType::CLAMP,
                    SamplerGLInterpType::LINEAR)
    , gradientTexture(gradientTexture)
    , currentTexture(GuideDebugGUIFuncs::PG_TEXTURE_SIZE, PixelFormat::RGB8_UNORM)
    , currentValues(GuideDebugGUIFuncs::PG_TEXTURE_SIZE[0] * GuideDebugGUIFuncs::PG_TEXTURE_SIZE[1], 0.0f)
    , maxValueDisplay(0.0f)
    , treeBuffer(0)
    , treeBufferSize(0)
{
    // Load the Name
    name = config[GuideDebug::NAME];
    // Load SDTrees to memory
    octrees.resize(depthCount);
    for(uint32_t i = 0; i < depthCount; i++)
    {
        LoadOctree(octrees[i], config, configPath, i);
    }
    // All done!
}

GDebugRendererSVO::~GDebugRendererSVO()
{
}

bool GDebugRendererSVO::LoadOctree(SVOctree& sdTree,
                                   const nlohmann::json& config,
                                   const std::string& configPath,
                                   uint32_t depth)
{
    auto loc = config.find(SVO_TREE_NAME);
    if(loc == config.end()) return false;
    if(depth >= loc->size()) return false;

    std::string fileName = (*loc)[depth];
    std::string fileMergedPath = Utility::MergeFileFolder(configPath, fileName);
    std::ifstream file(fileMergedPath, std::ios::binary);
    if(!file.good()) return false;

    static_assert(sizeof(char) == sizeof(Byte), "\"Byte\" is not have sizeof(char)");
    // Read STree Start Offset
    uint64_t sTreeOffset;
    file.read(reinterpret_cast<char*>(&sTreeOffset), sizeof(uint64_t));

    // Read SVO Buffers in order

    return true;
}

void GDebugRendererSVO::RenderSpatial(TextureGL& overlayTex, uint32_t depth,
                                      const std::vector<Vector3f>& worldPositions)
{
    // Calculate on CPU
    std::vector<Byte> falseColors(4 * overlayTex.Size().Multiply());
    // Generate iota array for parallel process
    std::vector<uint32_t> indices(overlayTex.Size().Multiply());
    std::iota(indices.begin(), indices.end(), 0);
    //
    std::for_each(std::execution::par_unseq,
                  indices.cbegin(), indices.cend(),
                  [&](uint32_t index)
                  {
                      Vector3f pos = worldPositions[index];
                      const SVOctree& svo = octrees[depth];

                      //
                      uint32_t svoLeafIndex;
                      bool found = svo.LeafIndex(svoLeafIndex, pos, true);
                      Vector3f locColor = (found) ? Utility::RandomColorRGB(svoLeafIndex)
                                                  : Vector3f(0.0f);

                      Vector4uc color;
                      color[0] = static_cast<uint8_t>(locColor[0] * 255.0f);
                      color[1] = static_cast<uint8_t>(locColor[1] * 255.0f);
                      color[2] = static_cast<uint8_t>(locColor[2] * 255.0f);
                      color[3] = 255;

                      falseColors[index * 4 + 0] = color[0];
                      falseColors[index * 4 + 1] = color[1];
                      falseColors[index * 4 + 2] = color[2];
                      falseColors[index * 4 + 3] = color[3];
                  });

    // Copy Data to the texture
    overlayTex.CopyToImage(falseColors, Zero2ui, overlayTex.Size(),
                           PixelFormat::RGBA8_UNORM);
}

void GDebugRendererSVO::UpdateDirectional(const Vector3f& worldPos,
                                          bool doLogScale,
                                          uint32_t depth)
{
    // Calculate on CPU
    currentValues.resize(mapSize.Multiply());
    // Generate iota array for parallel process
    std::vector<uint32_t> indices(mapSize.Multiply());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par_unseq,
                  indices.cbegin(), indices.cend(),
                  [&](uint32_t index)
                  {
                      Vector3f pos = worldPos;
                      const SVOctree& svo = octrees[depth];

                      // Calculate Direction

                      RayF ray;// (worldPos);

                      uint32_t leafIndex;
                      svo.TraceRay(leafIndex, ray, svo.leafVoxelSize * 0.5f,
                                   std::numeric_limits<float>::max());

                      float radiance = svo.ReadRadiance(leafIndex, true, -ray.getDirection());

                      currentValues[index] = radiance;
                  });
}

bool GDebugRendererSVO::RenderGUI(bool& overlayCheckboxChanged,
                                  bool& overlayValue,
                                  const ImVec2& windowSize)
{
    bool changed = false;
    using namespace GuideDebugGUIFuncs;

    ImGui::BeginChild(("##" + name).c_str(), windowSize, false);
    ImGui::SameLine(0.0f, CenteredTextLocation(name.c_str(), windowSize.x));
    overlayCheckboxChanged = ImGui::Checkbox("##OverlayCheckbox", &overlayValue);
    ImGui::SameLine();
    ImGui::Text("%s", name.c_str());
    ImVec2 remainingSize = FindRemainingSize(windowSize);
    remainingSize.x = remainingSize.y;
    ImGui::NewLine();
    ImGui::SameLine(0.0f, (windowSize.x - remainingSize.x) * 0.5f - ImGui::GetStyle().WindowPadding.x);
    RenderImageWithZoomTooltip(currentTexture, currentValues, remainingSize);

    if(ImGui::BeginPopupContextItem(("texPopup" + name).c_str()))
    {

        ImGui::Text("Resolution: [%u, %u]", mapSize[1], mapSize[0]);
        ImGui::Text("Max Value: %f", maxValueDisplay);
        ImGui::EndPopup();

    }
    ImGui::EndChild();
    return changed;
}