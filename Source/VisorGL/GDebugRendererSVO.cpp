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
#include "RayLib/FileUtility.h"
#include "RayLib/MortonCode.h"

#include "TextureGL.h"
#include "GuideDebugStructs.h"
#include "GuideDebugGUIFuncs.h"
#include "GLConversionFunctions.h"

// TODO: dont copy these
template <int32_t N>
class StaticGaussianFilter1D
{
    static_assert(N % 2 == 1, "Filter kernel size must be odd");
    public:
    static constexpr Vector2i   KERNEL_RANGE = Vector2i(-N / 2, N / 2);
    private:
    float                       kernel[N];

    public:
                                StaticGaussianFilter1D(float alpha);
    float                       operator()(int32_t i) const;
};

template <int32_t N>
StaticGaussianFilter1D<N>::StaticGaussianFilter1D(float alpha)
{
    auto Gauss = [&](float t)
    {
        constexpr float W = 1.0f / MathConstants::SqrtPi / MathConstants::Sqrt2;
        return W / std::sqrt(alpha) * std::exp(-(t * t) * 0.5f / alpha);
    };
    // Generate weights
    for(int i = KERNEL_RANGE[0]; i <= KERNEL_RANGE[1]; i++)
    {
        kernel[i + N / 2] = Gauss(static_cast<float>(i));
    };

    // Normalize the Kernel
    float total = 0.0f;
    for(int i = 0; i < N; i++) { total += kernel[i]; }
    for(int i = 0; i < N; i++) { kernel[i] /= total; }
}

template <int32_t N>
float StaticGaussianFilter1D<N>::operator()(int32_t i) const
{
    return kernel[i + (N / 2)];
}

Vector3f DirIdToWorldDir(const Vector2ui& dirXY,
                         const Vector2ui& dimensions)
{
    assert(dirXY < dimensions);
    using namespace MathConstants;
    // Spherical coordinate deltas
    Vector2f deltaXY = Vector2f((2.0f * Pi) / static_cast<float>(dimensions[0]),
                                Pi / static_cast<float>(dimensions[1]));

    // Assume image space bottom left is (0,0)
    // Center to the pixel as well
    Vector2f dirXYFloat = Vector2f(dirXY[0], dirXY[1]) + Vector2f(0.5f);
    Vector2f sphrCoords = Vector2f(-Pi + dirXYFloat[0] * deltaXY[0],
                                   Pi - dirXYFloat[1] * deltaXY[1]);
    Vector3f result = Utility::SphericalToCartesianUnit(sphrCoords);
    // Spherical Coords calculates as Z up change it to Y up
    Vector3 dirYUp = Vector3(result[1], result[2], result[0]);

    //METU_LOG("Pixel [{}, {}], ThetaPhi [{}, {}], Dir[{}, {}, {}]",
    //         dirXY[0], dirXY[1],
    //         sphrCoords[0] * RadToDegCoef,
    //         sphrCoords[1] * RadToDegCoef,
    //         dirYUp[0], dirYUp[1], dirYUp[2]);

    return dirYUp;
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

float SVOctree::ConeTraceRay(bool& isLeaf, uint32_t& leafId, const RayF& ray,
                             float tMin, float tMax, float coneAperture,
                             uint32_t maxQueryLevelOffset) const
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
        return std::exp2f(-static_cast<float>(power));
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

    // Aperture Related Routines
    const float CONE_DIAMETER_FACTOR = tan(0.5f * coneAperture) * 2.0f;
    auto ConeDiameter = [&CONE_DIAMETER_FACTOR](float distance)
    {
        return CONE_DIAMETER_FACTOR * distance;
    };
    auto LevelVoxelSize = [this](uint32_t currentLevel)
    {
        float levelFactor = static_cast<float>(1 << (leafDepth - currentLevel));
        return levelFactor * leafVoxelSize;
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
    // create cleaner code.
    // Initially, there should be at least one node
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

                // Check if the current cone aperture is larger than the
                // current voxel size, then terminate
                uint32_t voxelLevel = stack3BitCount + 1;
                bool isTerminated = ConeDiameter(tMin) > LevelVoxelSize(stack3BitCount + 1);
                isTerminated |= (static_cast<uint32_t>(leafDepth) - voxelLevel) <= maxQueryLevelOffset;
                bool isChildLeaf = IsChildrenLeaf(node);
                // If it is leaf just return the
                // "nodeId" it is actually leaf id
                if(isChildLeaf || isTerminated)
                {
                    isLeaf = isChildLeaf;
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

float SVOctree::NodeVoxelSize(uint32_t nodeIndex, bool isLeaf) const
{
    if(isLeaf) return leafVoxelSize;

    auto it = std::upper_bound(levelNodeOffsets.cbegin(), levelNodeOffsets.cend(),
                               nodeIndex);
    uint32_t distance = static_cast<uint32_t>(std::distance(levelNodeOffsets.cbegin(), it));

    uint32_t levelDiff = leafDepth - distance + 1;
    float multiplier = static_cast<float>(1 << levelDiff);
    return leafVoxelSize * multiplier;
}

bool SVOctree::Descend(uint32_t& index, uint64_t mortonCode,
                       uint32_t levelCap) const
{
    // Useful constants
    static constexpr uint32_t DIMENSION = 3;
    static constexpr uint32_t DIM_MASK = (1 << DIMENSION) - 1;

    uint32_t mortonLevelShift = (leafDepth - 1) * DIMENSION;
    // Now descend down
    uint32_t currentNodeIndex = 0;
    for(uint32_t i = 0; i < levelCap; i++)
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

bool SVOctree::NodeIndex(uint32_t& index, const Vector3f& worldPos,
                         uint32_t levelCap, bool checkNeighbours) const
{
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
        uint64_t voxelMorton = MortonCode::Compose<uint32_t>(voxIndex);
        // Traverse this morton code
        found = Descend(index, voxelMorton, levelCap);
        // Terminate if we are only checking a single voxel
        // or a voxel is found
        if(!checkNeighbours || found) break;
    }
    return found;
}

Vector3f SVOctree::ReadNormalAndSpecular(float& stdDev, float& specularity,
                                         uint32_t nodeIndex, bool isLeaf) const
{
    const uint32_t* gFetchArray = (isLeaf) ? normalAndSpecLeaf.data() : normalAndSpecNode.data();
    uint32_t packedData = gFetchArray[nodeIndex];

    Vector3f normal;
    // [0, 511]
    normal[0] = static_cast<float>((packedData >> NORMAL_X_OFFSET) & NORMAL_X_BIT_MASK);
    normal[1] = static_cast<float>((packedData >> NORMAL_Y_OFFSET) & NORMAL_Y_BIT_MASK);
    // [0, 2]
    normal[0] *= UNORM_NORM_X_FACTOR * 2.0f;
    normal[1] *= UNORM_NORM_Y_FACTOR * 2.0f;
    // [-1, 1]
    normal[0] -= 1.0f;
    normal[1] -= 1.0f;
    // Z axis from normalization
    normal[2] = sqrtf(std::max(0.0f, 1.0f - normal[0] * normal[0] - normal[1] * normal[1]));
    normal.NormalizeSelf();

    bool normalZSign = (packedData >> NORMAL_SIGN_BIT_OFFSET) & NORMAL_SIGN_BIT_MASK;
    normal[2] *= (normalZSign) ? -1.0f : 1.0f;

    float normalLength = static_cast<float>((packedData >> NORMAL_LENGTH_OFFSET) & NORMAL_LENGTH_BIT_MASK);
    normalLength *= UNORM_LENGTH_FACTOR;

    specularity = static_cast<float>((packedData >> SPECULAR_OFFSET) & SPECULAR_BIT_MASK);
    specularity *= UNORM_SPEC_FACTOR;

    // Toksvig 2004
    stdDev = (1.0f - normalLength) / normalLength;

    return normal;
}

float SVOctree::ReadRadiance(const Vector3f& coneDirection,
                             float coneAperture, uint32_t nodeIndex,
                             bool isLeaf) const
{
    // Debug Visualizer does not have access to the boundary light
    // just return zero here
    if(nodeIndex == UINT32_MAX) return 0.0f;

    const Vector2f* gIrradArray = (isLeaf) ? avgIrradianceLeaf.data() : avgIrradianceNode.data();

    float normalDeviation, specularity;
    // Fetch normal to estimate the surface confidence
    Vector3f normal = ReadNormalAndSpecular(normalDeviation, specularity,
                                            nodeIndex, isLeaf);

    // TODO: Implement radiance distribution incorporation
    // Generate a Gaussian using the bit quadtree for outgoing radiance
    // Incorporate normal, (which also is a gaussian)
    // Convolve all these three to generate an analytic function (freq domain)
    // All functions are gaussians so fourier transform is analytic
    // Convert it back
    // Sample the value using tetha (coneDir o normal)

    // Currently we are just fetching irradiance
    // which side are we on
    // Note that "coneDirection" is towards to the surface
    bool towardsNormal = (normal.Dot(-coneDirection) >= 0.0f);
    uint32_t index = towardsNormal ? 0 : 1;

    return gIrradArray[nodeIndex][index];
}

GDebugRendererSVO::GDebugRendererSVO(const nlohmann::json& config,
                                     const TextureGL& gradientTexture,
                                     const std::string& configPath,
                                     uint32_t depthCount)
    : linearSampler(SamplerGLEdgeResolveType::CLAMP,
                    SamplerGLInterpType::LINEAR)
    , gradientTexture(gradientTexture)
    , compReduction(ShaderType::COMPUTE, u8"Shaders/TextureMaxReduction.comp")
    , compRefRender(ShaderType::COMPUTE, u8"Shaders/PGReferenceRender.comp")
    , maxValueDisplay(0.0f)
{
    // Load the Name
    name = config[GuideDebug::NAME];
    mapSize = SceneIO::LoadVector<2, uint32_t>(config[MAP_SIZE_NAME]);

    minBinLevel = config[TRACE_LEVEL_NAME];

    // Load Exact Sized Texture
    currentTexture = TextureGL(mapSize, PixelFormat::RGBA8_UNORM);
    currentValues.resize(mapSize.Multiply());

    // Preload some integer names for bin level selection
    // 2^16 x 2^16 x 2^16  SVO should be impossible
    // TODO: maybe change later
    for(int i = 0; i < 16; i++)
    {
        nameList.push_back(std::make_pair(i, std::to_string(i)));
    }

    // Load SDTrees to memory
    octrees.resize(depthCount);
    for(uint32_t i = 0; i < depthCount; i++)
    {
        LoadOctree(octrees[i], config, configPath, i);
    }
    currentIndex = minBinLevel;
    // All done!
}

GDebugRendererSVO::~GDebugRendererSVO()
{
}

bool GDebugRendererSVO::LoadOctree(SVOctree& octree,
                                   const nlohmann::json& config,
                                   const std::string& configPath,
                                   uint32_t depth)
{
    auto loc = config.find(SVO_TREE_NAME);
    if(loc == config.end()) return false;
    if(depth >= loc->size()) return false;

    std::string fileName = (*loc)[depth];
    std::string fileMergedPath = Utility::MergeFileFolder(configPath, fileName);
    static_assert(sizeof(char) == sizeof(Byte), "\"Byte\" is not have sizeof(char)");

    METU_LOG("Loading: {:s}", fileMergedPath);

    std::vector<Byte> data;
    Utility::DevourFileToStdVector(data, fileMergedPath);

    // Init Zeroes
    octree = {};
    // Start by fetching the "header"
    size_t offset = 0;
    std::memcpy(&octree.svoAABB, data.data() + offset, sizeof(AABB3f));
    offset += sizeof(AABB3f);
    std::memcpy(&octree.voxelResolution, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(&octree.leafDepth, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(&octree.nodeCount, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(&octree.leafCount, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    std::memcpy(&octree.leafVoxelSize, data.data() + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&octree.levelOffsetCount, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // After that generate the sizes
    std::array<size_t, 5> treeSizes = {};
    // Actual Tree Related Sizes
    // Node Amount
    treeSizes[0] = octree.nodeCount * sizeof(uint64_t);       // "dNodes" Size
    treeSizes[1] = octree.nodeCount * sizeof(uint16_t);        // "dBinInfo" Size
    // Leaf Amount
    treeSizes[2] = octree.leafCount * sizeof(uint32_t);        // "dLeafParents" Size
    treeSizes[3] = octree.leafCount * sizeof(uint16_t);        // "dLeafBinInfo" Size
    // Misc
    treeSizes[4] = octree.levelOffsetCount * sizeof(uint32_t); // "dLevelNodeOffsets" Size

    // Payload Related Sizes
    std::array<size_t, 10> payloadSizes = {};
    // Node Amount
    payloadSizes[0] = octree.nodeCount * sizeof(Vector2f);     // "dAvgIrradianceNode" Size
    payloadSizes[1] = octree.nodeCount * sizeof(uint32_t);     // "dNormalAndSpecNode" Size
    payloadSizes[2] = octree.nodeCount * sizeof(uint8_t);      // "dGuidingFactorNode" Size
    //payloadSizes[3] = octree.nodeCount * sizeof(uint64_t);   // "dMicroQuadTreeNode" Size
    //Leaf Amount
    payloadSizes[4] = octree.leafCount * sizeof(Vector2f);    // "dTotalIrradianceLeaf" Size
    payloadSizes[5] = octree.leafCount * sizeof(Vector2ui);    // "dSampleCountLeaf" Size
    payloadSizes[6] = octree.leafCount * sizeof(Vector2f);     // "dAvgIrradianceLeaf" Size
    payloadSizes[7] = octree.leafCount * sizeof(uint32_t);     // "dNormalAndSpecLeaf" Size
    payloadSizes[8] = octree.leafCount * sizeof(uint8_t);      // "dGuidingFactorLeaf" Size
    //payloadSizes[9] = octree.leafCount * sizeof(uint64_t);   // "dMicroQuadTreeNode" Size

    // Calculate the offsets and total size
    size_t treeTotalSize = std::reduce(treeSizes.cbegin(), treeSizes.cend(), 0ull);
    size_t payloadTotalSize = std::reduce(payloadSizes.cbegin(), payloadSizes.cend(), 0ull);

    // Sanity check the calculated size
    size_t totalSize = (treeTotalSize + payloadTotalSize +
                        sizeof(AABB3f) + 5 * sizeof(uint32_t) +
                        sizeof(float));
    if(totalSize != data.size()) return false;

    // Allocate
    octree.levelNodeOffsets.resize(octree.levelOffsetCount);
    octree.nodes.resize(octree.nodeCount);
    octree.binInfo.resize(octree.nodeCount);
    octree.leafParents.resize(octree.leafCount);
    octree.leafBinInfo.resize(octree.leafCount);
    octree.totalIrradianceLeaf.resize(octree.leafCount);
    octree.sampleCountLeaf.resize(octree.leafCount);
    octree.avgIrradianceLeaf.resize(octree.leafCount);
    octree.normalAndSpecLeaf.resize(octree.leafCount);
    octree.guidingFactorLeaf.resize(octree.leafCount);
    octree.avgIrradianceNode.resize(octree.nodeCount);
    octree.normalAndSpecNode.resize(octree.nodeCount);
    octree.guidingFactorNode.resize(octree.nodeCount);

    // Now copy stuff
    // Tree Related
    // "dNodes"
    std::memcpy(octree.nodes.data(), data.data() + offset, treeSizes[0]);
    offset += treeSizes[0];
    // "dBinInfo"
    std::memcpy(octree.binInfo.data(), data.data() + offset, treeSizes[1]);
    offset += treeSizes[1];
    // "dLeafParents"
    std::memcpy(octree.leafParents.data(), data.data() + offset, treeSizes[2]);
    offset += treeSizes[2];
    // "dLeafBinInfo"
    std::memcpy(octree.leafBinInfo.data(), data.data() + offset, treeSizes[3]);
    offset += treeSizes[3];
    // "dLevelNodeOffsets"
    std::memcpy(octree.levelNodeOffsets.data(), data.data() + offset, treeSizes[4]);
    offset += treeSizes[4];
    // Payload Related
    // "dAvgIrradianceNode"
    std::memcpy(octree.avgIrradianceNode.data(), data.data() + offset, payloadSizes[0]);
    offset += payloadSizes[0];
    // "dNormalAndSpecNode"
    std::memcpy(octree.normalAndSpecNode.data(), data.data() + offset, payloadSizes[1]);
    offset += payloadSizes[1];
    // "dGuidingFactorNode"
    std::memcpy(octree.guidingFactorNode.data(), data.data() + offset, payloadSizes[2]);
    offset += payloadSizes[2];
    //// "dMicroQuadTreeNode"
    //std::memcpy(octree.microQuadTreeNode.data(), data.data() + offset, payloadSizes[3]);
    //offset += payloadSizes[3];
    // "dTotalIrradianceLeaf"
    std::memcpy(octree.totalIrradianceLeaf.data(), data.data() + offset, payloadSizes[4]);
    offset += payloadSizes[4];
    // "dTotalIrradianceLeaf"
    std::memcpy(octree.sampleCountLeaf.data(), data.data() + offset, payloadSizes[5]);
    offset += payloadSizes[5];
    // "dSampleCountLeaf"
    std::memcpy(octree.avgIrradianceLeaf.data(), data.data() + offset, payloadSizes[6]);
    offset += payloadSizes[6];
    // "dNormalAndSpecLeaf"
    std::memcpy(octree.normalAndSpecLeaf.data(), data.data() + offset, payloadSizes[7]);
    offset += payloadSizes[7];
    // "dGuidingFactorLeaf"
    std::memcpy(octree.guidingFactorLeaf.data(), data.data() + offset, payloadSizes[8]);
    offset += payloadSizes[8];
    //// "dMicroQuadTreeLeaf"
    //std::memcpy(octree.microQuadTreeLeaf.data(), data.data() + offset, payloadSizes[9]);
    //offset += payloadSizes[9];
    // All Done!
    assert(offset == data.size());
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
                      uint32_t svoNodeIndex;
                      bool found = svo.NodeIndex(svoNodeIndex, pos,
                                                 std::min(minBinLevel, svo.leafDepth),
                                                 true);
                      Vector3f locColor = (found) ? Utility::RandomColorRGB(svoNodeIndex)
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
    auto ConeAperture = [](Vector2ui mapSize) -> float
    {
        // Octo mapping has equal area so directly divide
        static constexpr float MAX_STERAD = 4.0f * MathConstants::Pi;
        uint32_t totalPix = mapSize.Multiply();
        return MAX_STERAD / static_cast<float>(totalPix);
    };
    const SVOctree& svo = octrees[depth];


    //
    Vector3f pos = worldPos;
    // Convert location to the
    uint32_t nodeIndex;
    bool found = svo.NodeIndex(nodeIndex, pos,
                               std::min(minBinLevel, svo.leafDepth),
                               true);
    if(!found) METU_ERROR_LOG("Unable to locate a voxel! Using direct world space for ray position!");

    // Now find out the node id and offset the tmin accordingly
    float voxSize = svo.NodeVoxelSize(nodeIndex, (minBinLevel == svo.leafDepth));
    float tMin = voxSize * MathConstants::Sqrt3 + MathConstants::LargeEpsilon;

    float coneAperture = ConeAperture(mapSize);
    currentValues.resize(mapSize.Multiply());
    // Generate iota array for parallel process
    std::vector<uint32_t> indices(mapSize.Multiply());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par_unseq,
                  indices.cbegin(), indices.cend(),
                  [&](uint32_t index)
                  {
                      // Calculate Direction
                      Vector2ui pixelId(index % mapSize[0],
                                        index / mapSize[0]);
                      Vector3f direction = DirIdToWorldDir(pixelId, mapSize);
                      RayF ray(direction, pos);

                      bool isLeaf;
                      uint32_t leafIndex;
                      svo.ConeTraceRay(isLeaf, leafIndex, ray, tMin,
                                       std::numeric_limits<float>::max(),
                                       coneAperture);

                      float radiance = svo.ReadRadiance(ray.getDirection(), coneAperture,
                                                        leafIndex, isLeaf);
                      currentValues[index] = radiance;
                  });


    std::vector<float> valuesBuffer(currentValues.size(), 0.0f);
    using FilterType = StaticGaussianFilter1D<5>;
    FilterType gaussFilter(1.0f);
    auto KERNEL_RANGE = FilterType::KERNEL_RANGE;
    // Do Gauss Pass over Values
    std::for_each(std::execution::par_unseq,
                  indices.cbegin(), indices.cend(),
                  [&](uint32_t index)
                  {
                      // Calculate Direction
                      Vector2i pixelId(index % mapSize[0],
                                       index / mapSize[0]);

                      for(int32_t j = KERNEL_RANGE[0]; j <= KERNEL_RANGE[1]; ++j)
                      {
                          int32_t row = pixelId[1];
                          int32_t col = HybridFuncs::Clamp<int32_t>(pixelId[0] + j, 0, mapSize[0] - 1);
                          int32_t linear = row * mapSize[0] + col;
                          valuesBuffer[index] += currentValues[linear] * gaussFilter(j);
                      }
                  });
    currentValues.resize(currentValues.size(), 0.0f);
    std::for_each(std::execution::par_unseq,
                  indices.cbegin(), indices.cend(),
                  [&](uint32_t index)
                  {
                      // Calculate Direction
                      Vector2i pixelId(index % mapSize[0],
                                       index / mapSize[0]);

                      for(int32_t j = KERNEL_RANGE[0]; j <= KERNEL_RANGE[1]; ++j)
                      {
                          int32_t row = HybridFuncs::Clamp<int32_t>(pixelId[1] + j, 0, mapSize[1] - 1);
                          int32_t col = pixelId[0];
                          int32_t linear = row * mapSize[0] + col;
                          currentValues[index] += valuesBuffer[linear] * gaussFilter(j);
                      }
                  });

    // Normalize the indices
    maxValueDisplay = std::reduce(std::execution::par_unseq,
                                  currentValues.cbegin(), currentValues.cend(),
                                  -std::numeric_limits<float>::max(),
                                  [](float a, float b) { return std::max(a, b); });

    // Copy the actual current values to a byte buffer to copy....
    std::vector<Byte> tempCurValsForCopy;
    tempCurValsForCopy.resize(mapSize.Multiply() * sizeof(float));
    std::memcpy(tempCurValsForCopy.data(), currentValues.data(),
                mapSize.Multiply() * sizeof(float));

    // Load temporarily to a texture
    TextureGL curTexture = TextureGL(mapSize, PixelFormat::R_FLOAT);
    curTexture.CopyToImage(tempCurValsForCopy, Zero2ui, mapSize, PixelFormat::R_FLOAT);

    // ============================= //
    //     Call Reduction Shader     //
    // ============================= //
    // Get a max luminance buffer;
    float initalMaxData = 0.0f;
    GLuint maxBuffer;
    glGenBuffers(1, &maxBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, maxBuffer);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, 1 * sizeof(float), &initalMaxData, 0);

    // Both of these compute shaders total work count is same
    const GLuint workCount = curTexture.Size()[1] * curTexture.Size()[0];
    // Some WG Definitions (statically defined in shader)
    static constexpr GLuint WORK_GROUP_1D_X = 256;
    static constexpr GLuint WORK_GROUP_2D_X = 16;
    static constexpr GLuint WORK_GROUP_2D_Y = 16;
    // =======================================================
    // Set Max Shader
    compReduction.Bind();
    // Bind Uniforms
    glUniform2ui(U_RES, curTexture.Size()[0], curTexture.Size()[1]);
    // Bind SSBO
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSB_MAX_LUM, maxBuffer);
    // Textures
    curTexture.Bind(T_IN_LUM_TEX);
    // Dispatch Max Shader
    // Max shader is 1D shader set data accordingly
    GLuint gridX_1D = (workCount + WORK_GROUP_1D_X - 1) / WORK_GROUP_1D_X;
    glDispatchCompute(gridX_1D, 1, 1);
    glMemoryBarrier(GL_UNIFORM_BARRIER_BIT |
                    GL_SHADER_STORAGE_BARRIER_BIT);
    // =======================================================
    // Unbind SSBO just to be sure
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSB_MAX_LUM, 0);
    // ============================= //
    //     Call Reduction Shader     //
    // ============================= //
    // Set Render Shader
    compRefRender.Bind();
    // Bind Uniforms
    glUniform2ui(U_RES, curTexture.Size()[0], curTexture.Size()[1]);
    glUniform1i(U_LOG_ON, doLogScale ? 1 : 0);
    //
    // UBOs
    glBindBufferBase(GL_UNIFORM_BUFFER, UB_MAX_LUM, maxBuffer);
    // Textures
    curTexture.Bind(T_IN_LUM_TEX);
    gradientTexture.Bind(T_IN_GRAD_TEX);  linearSampler.Bind(T_IN_GRAD_TEX);
    // Images
    glBindImageTexture(I_OUT_REF_IMAGE, currentTexture.TexId(),
                       0, false, 0, GL_WRITE_ONLY,
                       PixelFormatToSizedGL(currentTexture.Format()));
    // Dispatch Render Shader
    // Max shader is 2D shader set data accordingly
    GLuint gridX_2D = (curTexture.Size()[0] + WORK_GROUP_2D_X - 1) / WORK_GROUP_2D_X;
    GLuint gridY_2D = (curTexture.Size()[1] + WORK_GROUP_2D_Y - 1) / WORK_GROUP_2D_Y;
    glDispatchCompute(gridX_2D, gridY_2D, 1);
    // =======================================================
    // All done!!!

    // Delete Temp Max Buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glDeleteBuffers(1, &maxBuffer);

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
        ImGui::Text("BinLevel"); ImGui::SameLine();
        if(ImGui::BeginCombo("##BinLevelCombo", nameList[currentIndex].second.c_str()))
        {
            // TODO: assuming all trees are the same here change if necessary
            for(uint32_t i = 1; i <= octrees[0].leafDepth; i++)
            {
                bool isSelected = (currentIndex == i);
                if(ImGui::Selectable(nameList[i].second.c_str(), isSelected))
                    currentIndex = i;
                if(isSelected)
                {

                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::EndPopup();

    }
    ImGui::EndChild();

    if(minBinLevel != currentIndex)
    {
        minBinLevel = currentIndex;
        overlayCheckboxChanged = true;
        changed = true;
    }

    return changed;
}