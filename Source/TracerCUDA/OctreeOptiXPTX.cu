#include  "OctreeOptiXPTX.cuh"
#include "RayLib/RandomColor.h"
#include "WFPGTracerKC.cuh"

#include <optix_device.h>

extern "C" __constant__ OctreeAccelParams params;

// Meta Closest Hit Shader
__device__ __forceinline__
void KCClosestHitSVO()
{
    const int leafId = optixGetPrimitiveIndex();
    optixSetPayload_1(leafId);
    optixSetPayload_2(__float_as_uint(optixGetRayTmax()));
}

__device__ __forceinline__
void KCMissSVOOptiX()
{
    // Do Nothing
}

template <class T>
__device__ __forceinline__
void KCIntersectVoxel()
{
    // TODO: Docs says object space is faster?
    // check it
    float3 rayOrig = optixGetWorldRayOrigin();
    float3 rayDir = optixGetWorldRayDirection();
    uint32_t currentLevel = optixGetPayload_0();

    const int leafId = optixGetPrimitiveIndex();
    const T* dMortonCodes = *(const T**)(optixGetSbtDataPointer());

    // Although AABB == the Voxel, we cant query the hit tMin
    // from the API (probably it does not have it)
    // Do the intersection "by hand"
    RayF ray = RayF(Vector3f(rayDir.x, rayDir.y, rayDir.z),
                    Vector3f(rayOrig.x, rayOrig.y, rayOrig.z));
    Vector2f tMinMax = Vector2f(optixGetRayTmin(),
                                optixGetRayTmax());
    // Generate the AABB
    // "Decompress the Morton code to AABB"
    AABB3f octreeAABB = params.svo.OctreeAABB();
    float levelVoxSize = params.svo.LevelVoxelSize(currentLevel);
    Vector3ui voxId = MortonCode::Decompose3D<T>(dMortonCodes[leafId]);
    Vector3f voxIdF = Vector3f(voxId);
    Vector3f voxAABBMin = octreeAABB.Min() + voxIdF * levelVoxSize;
    Vector3f voxAABBMax = voxAABBMin + levelVoxSize;
    // Actual AABB intersection
    float newT;
    Vector3f position;
    if(ray.IntersectsAABB(position, newT,
                          voxAABBMin,
                          voxAABBMax,
                          tMinMax))
    {
        optixReportIntersection(newT, 0);
    }
}

//__device__ __forceinline__
//void KCCamTraceSVO()
//{
//    // Sanity Check (for payload)
//    static_assert(sizeof(uint32_t) == sizeof(float));
//    // We Launch linearly
//    const uint32_t launchDim = optixGetLaunchDimensions().x;
//    const uint32_t launchIndex = optixGetLaunchIndex().x;
//    // Should we check this ??
//    if(launchIndex >= launchDim) return;
//    // Camera Rays are generated from other kernel
//    RayReg ray = RayReg(params.gRays, launchIndex);
//
//    // Trace hierarchically and find the optimal location
//    uint32_t foundLevel = UINT32_MAX;
//    uint32_t nodeRelativeIndex = UINT32_MAX;
//    float currentTMin = ray.tMin;
//
//    const AnisoSVOctreeGPU& svo = params.svo;
//    int endLevel = static_cast<int>(svo.LeafDepth() - params.maxQueryOffset);
//    endLevel = max(1, endLevel);
//    // We don't have root SVO skip it
//    for(int i = endLevel; i <= endLevel; i++)
//    {
//        int accIndex = (i - 1);
//        // Ray Payload
//        // In
//        uint32_t currentLevel = static_cast<uint32_t>(i);
//        // Out
//        uint32_t localHitNodeIndex = UINT32_MAX;
//        uint32_t tMinOutUInt32 = __float_as_uint(FLT_MAX);
//
//        // Trace Call
//        optixTrace(// Accelerator
//                   params.octreeLevelBVHs[accIndex],
//                   // Ray Input
//                   make_float3(ray.ray.getPosition()[0],
//                               ray.ray.getPosition()[1],
//                               ray.ray.getPosition()[2]),
//                   make_float3(ray.ray.getDirection()[0],
//                               ray.ray.getDirection()[1],
//                               ray.ray.getDirection()[2]),
//                   currentTMin,
//                   ray.tMax,
//                   0.0f,
//                   //
//                   OptixVisibilityMask(255),
//                   // Flags
//                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//                   // SBT
//                   accIndex, 1, 0,
//                   // Payload
//                   currentLevel,
//                   localHitNodeIndex,
//                   tMinOutUInt32);
//
//        // Fetch the found tMin
//        float tMinOut = __uint_as_float(tMinOutUInt32);
//
//        // We missed the current level SVO
//        // Last saved valid location is hit
//        if(tMinOut == FLT_MAX)
//        {
//            //if(curr)
//            //printf("missed!\n");
//            break;
//        }
//
//        // Accept hit if new hit is closer
//        // Accept it even if it is the same
//        if(tMinOut >= currentTMin)
//        {
//            // By SVO structure (AABB is convex shape and
//            // each refined resolution guarantees that found tMin
//            // will be larger that the higher level's tMin)
//            // we can optimize the traversal
//            currentTMin = tMinOut;//, FLT_MAX);
//
//            // We have a hit if code flows here
//            // Check if the hit location is valid
//            // (cone aperture vs. voxel size check)
//            float distance = tMinOut - ray.tMin;
//            float area = distance * distance * params.pixelAperture;
//            // Approximate with a circle
//            float coneDiskRadusSqr = area * MathConstants::InvPi;
//            float coneDiskDiamSqr = 4.0f * coneDiskRadusSqr;
//
//            float levelVoxelSize = svo.LevelVoxelSize(currentLevel);
//            levelVoxelSize = levelVoxelSize * levelVoxelSize;
//
//            // Voxel is just good enough than cone aperture at that range
//            // Voxel is larger accept the hit
//            // Next loop may refine it later
//            if(coneDiskDiamSqr <= levelVoxelSize)
//            {
//                foundLevel = currentLevel;
//                nodeRelativeIndex = localHitNodeIndex;
//            }
//            // If disk radius is larger than the voxel size
//            // break the loop. Since voxel size will get smaller
//            // and disk radius may stay the same or get larger
//            //else break;
//        }
//    }
//
//    // Finally Query the result
//    bool isLeaf = (foundLevel == svo.LeafDepth());
//    uint32_t globalNodeId = nodeRelativeIndex;
//    if(nodeRelativeIndex != UINT32_MAX && !isLeaf)
//        globalNodeId += svo.LevelNodeStart(foundLevel);
//
//    // Octree Display Mode
//    Vector3f rayDir = ray.ray.getDirection();
//    Vector4f locColor = Vector4f(0.0f, 0.0f, 10.0f, 1.0f);
//
//    WFPGRenderMode mode = params.renderMode;
//    if(mode == WFPGRenderMode::SVO_FALSE_COLOR)
//        locColor = (globalNodeId != UINT32_MAX)
//            ? Vector4f(Utility::RandomColorRGB(globalNodeId), 1.0f)
//            : Vector4f(Vector3f(0.0f), 1.0f);
//    // Payload Display Mode
//    else if(globalNodeId == UINT32_MAX)
//        locColor = Vector4f(1.0f, 0.0f, 1.0f, 1.0f);
//    else if(mode == WFPGRenderMode::SVO_RADIANCE)
//    {
//        //Vector3f hitPos = ray.ray.getPosition() + rayDir.Normalize() * tMin;
//        //float radianceF = ReadInterpolatedRadiance(hitPos, rayDir,
//        //                                           params.pixelAperture,
//        //                                           svo);
//
//        float radiance = svo.ReadRadiance(rayDir, params.pixelAperture,
//                                         globalNodeId, isLeaf);
//        float radianceF = radiance;
//        locColor = Vector4f(Vector3f(radianceF), 1.0f);
//    }
//    else if(mode == WFPGRenderMode::SVO_NORMAL)
//    {
//        float stdDev;
//        Vector3f normal = svo.DebugReadNormal(stdDev, globalNodeId, isLeaf);
//
//        // Voxels are two sided show the normal for the current direction
//        normal = (normal.Dot(rayDir) >= 0.0f) ? normal : -normal;
//
//        // Convert normal to 0-1 range
//        normal += Vector3f(1.0f);
//        normal *= Vector3f(0.5f);
//        locColor = Vector4f(normal, stdDev);
//    }
//
//    // Actual Write
//    uint32_t sampleIndex = params.gRayAux[launchIndex].sampleIndex;
//    params.gSamples.gValues[sampleIndex] = locColor;
//}

//__device__ __forceinline__
//void KCCamTraceSVO()
//{
//    // Sanity Check (for payload)
//    static_assert(sizeof(uint32_t) == sizeof(float));
//    // We Launch linearly
//    const uint32_t launchDim = optixGetLaunchDimensions().x;
//    const uint32_t launchIndex = optixGetLaunchIndex().x;
//    // Should we check this ??
//    if(launchIndex >= launchDim) return;
//    // Camera Rays are generated from other kernel
//    RayReg ray = RayReg(params.gRays, launchIndex);
//
//    const AnisoSVOctreeGPU& svo = params.svo;
//
//    // Find the upper bound voxel size
//    Vector2f tMinMaxOut;
//    bool skipsSVO = !ray.ray.IntersectsAABB(tMinMaxOut,
//                                            svo.OctreeAABB().Min(),
//                                            svo.OctreeAABB().Max(),
//                                            Vector2f(ray.tMin, ray.tMax));
//    float maxDiskDiamSqr = CalcConeDiskDiameterSqr(tMinMaxOut[1] - tMinMaxOut[0]);
//    float leavVoxSizeSqr = svo.LeafVoxelSize() * svo.LeafVoxelSize();
//    float level = log2(maxDiskDiamSqr / leavVoxSizeSqr) * 0.5f;
//    level = max(level, 0.0f);
//    int upperBound = static_cast<int>(floor(level));
//    // At most this will be the last check
//    // since voxel sizes will be larger after that
//    int endLevel = 1;// max(1, svo.LeafDepth() - upperBound);
//    // Start level is leaf or the offseted leaf
//    // (query offset is used  for debugging only)
//    int startLevel = static_cast<int>(svo.LeafDepth() - params.maxQueryOffset);
//    startLevel = max(1, startLevel);
//
//    // Trace hierarchically and find the optimal location
//    uint32_t foundLevel = UINT32_MAX;
//    uint32_t nodeRelativeIndex = UINT32_MAX;
//    float currentTMax = ray.tMax;
//
//    // We don't have root SVO skip it
//    for(int i = startLevel; i >= endLevel; i--)
//    {
//        int accIndex = (i - 1);
//        // Ray Payload
//        // In
//        uint32_t currentLevel = static_cast<uint32_t>(i);
//        // Out
//        uint32_t localHitNodeIndex = UINT32_MAX;
//        uint32_t tMaxOutUInt32 = __float_as_uint(currentTMax);
//
//        // Trace Call
//        optixTrace(// Accelerator
//                   params.octreeLevelBVHs[accIndex],
//                   // Ray Input
//                   make_float3(ray.ray.getPosition()[0],
//                               ray.ray.getPosition()[1],
//                               ray.ray.getPosition()[2]),
//                   make_float3(ray.ray.getDirection()[0],
//                               ray.ray.getDirection()[1],
//                               ray.ray.getDirection()[2]),
//                   ray.tMin,
//                   currentTMax,
//                   0.0f,
//                   //
//                   OptixVisibilityMask(255),
//                   // Flags
//                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//                   // SBT
//                   accIndex, 1, 0,
//                   // Payload
//                   currentLevel,
//                   localHitNodeIndex,
//                   tMaxOutUInt32);
//
//        // Fetch the found tMin
//        float tMaxOut = __uint_as_float(tMaxOutUInt32);
//
//        if(tMaxOut == currentTMax)
//        {
//            //printf("missed ");
//            continue;
//        }
//
//        // Accept hit if new hit is closer
//        if(tMaxOut <= currentTMax)
//        {
//            // We have a hit if code flows here
//            // Check if the hit location is valid
//            // (cone aperture vs. voxel size check)
//            float distance = tMaxOut - ray.tMin;
//            float coneDiskDiamSqr = CalcConeDiskDiameterSqr(distance);
//            float levelVoxelSize = svo.LevelVoxelSize(currentLevel);
//            float levelVoxelSizeSqr = levelVoxelSize * levelVoxelSize;
//            float dvRatio = log2(coneDiskDiamSqr / levelVoxelSizeSqr) * 0.5f;
//
//            // By SVO structure (AABB is convex shape and
//            // each refined resolution guarantees that found tMin
//            // will be larger that the higher level's tMin)
//            // we can optimize the traversal
//            currentTMax = tMaxOut + levelVoxelSize * 0.5f;
//
//            // Next iteration definitely will be larger
//            // By definition tMax can only be smaller
//            // meaning diameter will be smaller but voxel size will be larger
//            // accept this hit and break
//
//            // We are at leaf but already cone is smaller than voxel
//            // SVO resolution is not enough so accept the hit and terminate
//            bool leafTrace = (i == startLevel);
//            bool lowResSVO = (leafTrace) && (coneDiskDiamSqr < levelVoxelSizeSqr);
//            //bool properInterval = (coneDiskDiamSqr > levelVoxelSizeSqr &&
//            //                       coneDiskDiamSqr <= levelVoxelSizeNextSqr);
//            bool properInterval = (dvRatio > -0.5f && dvRatio < 0.5f);
//            if(lowResSVO || properInterval)
//            {
//                // Accept the hit and terminate
//                foundLevel = currentLevel;
//                nodeRelativeIndex = localHitNodeIndex;
//                break;
//            }
//        }
//    }
//
//    // Finally Query the result
//    bool isLeaf = (foundLevel == svo.LeafDepth());
//    uint32_t globalNodeId = nodeRelativeIndex;
//    if(nodeRelativeIndex != UINT32_MAX && !isLeaf)
//        globalNodeId += svo.LevelNodeStart(foundLevel);
//
//    // Octree Display Mode
//    Vector3f rayDir = ray.ray.getDirection();
//    Vector4f locColor = Vector4f(0.0f, 0.0f, 10.0f, 1.0f);
//
//    Vector3f asd[10] =
//    {
//        Vector3f(1.0f), Vector3f(1.0f), Vector3f(1.0f), Vector3f(1.0f),
//        Vector3f(1.0f), Vector3f(1.0f), Vector3f(1.0f), Vector3f(0.0f, 0.0f, 1.0f),
//        Vector3f(0.0f, 1.0f,0.0f) , Vector3f(1.0f,0.0f,0.0f)
//    };
//
//    WFPGRenderMode mode = params.renderMode;
//    if(mode == WFPGRenderMode::SVO_FALSE_COLOR)
//    {
//        // Saturate using level
//        Vector3f color = Utility::RandomColorRGB(globalNodeId);
//        color *= static_cast<float>(svo.LeafDepth()) / static_cast<float>(foundLevel);
//        locColor = (globalNodeId != UINT32_MAX)
//                        ? Vector4f(color, 1.0f)
//                        : Vector4f(Vector3f(0.0f), 1.0f);
//    }
//
//    // Payload Display Mode
//    else if(globalNodeId == UINT32_MAX)
//        locColor = Vector4f(1.0f, 0.0f, 1.0f, 1.0f);
//    else if(mode == WFPGRenderMode::SVO_RADIANCE)
//    {
//        //Vector3f hitPos = ray.ray.getPosition() + rayDir.Normalize() * currentTMax;
//        //float radianceF = ReadInterpolatedRadiance(hitPos, rayDir,
//        //                                           params.pixelAperture,
//        //                                           svo);
//
//        float radiance = svo.ReadRadiance(rayDir, params.pixelAperture,
//                                         globalNodeId, isLeaf);
//        float radianceF = radiance;
//        locColor = Vector4f(Vector3f(radianceF), 1.0f);
//    }
//    else if(mode == WFPGRenderMode::SVO_NORMAL)
//    {
//        float stdDev;
//        Vector3f normal = svo.DebugReadNormal(stdDev, globalNodeId, isLeaf);
//
//        // Voxels are two sided show the normal for the current direction
//        normal = (normal.Dot(rayDir) >= 0.0f) ? normal : -normal;
//
//        // Convert normal to 0-1 range
//        normal += Vector3f(1.0f);
//        normal *= Vector3f(0.5f);
//        locColor = Vector4f(normal, stdDev);
//    }
//
//    // Actual Write
//    uint32_t sampleIndex = params.gRayAux[launchIndex].sampleIndex;
//    params.gSamples.gValues[sampleIndex] = locColor;
//}

__device__ __forceinline__
void KCCamTraceSVO()
{
    // Sanity Check (for payload)
    static_assert(sizeof(uint32_t) == sizeof(float));
    // We Launch linearly
    const uint32_t launchDim = optixGetLaunchDimensions().x;
    const uint32_t launchIndex = optixGetLaunchIndex().x;
    // Should we check this ??
    if(launchIndex >= launchDim) return;
    // Camera Rays are generated from other kernel
    RayReg ray = RayReg(params.gRays, launchIndex);
    // Actual SVO structure will be used to expand the hit location wrt.
    // query level
    const AnisoSVOctreeGPU& svo = params.svo;

    // Start level is leaf or the offseted leaf
    // (query offset is used  for debugging only)
    int startLevel = static_cast<int>(svo.LeafDepth() - params.maxQueryOffset);
    startLevel = max(1, startLevel);
    int accIndex = (startLevel - 1);
    // Ray Payload
    // In
    uint32_t currentLevel = svo.LeafDepth();
    // Out
    uint32_t leafNodeIdOut = UINT32_MAX;
    uint32_t tMaxOutUInt32 = __float_as_uint(ray.tMax);

    // Trace Call
    optixTrace(// Accelerator
               params.octreeLevelBVHs[accIndex],
               // Ray Input
               make_float3(ray.ray.getPosition()[0],
                           ray.ray.getPosition()[1],
                           ray.ray.getPosition()[2]),
               make_float3(ray.ray.getDirection()[0],
                           ray.ray.getDirection()[1],
                           ray.ray.getDirection()[2]),
               ray.tMin,
               ray.tMax,
               0.0f,
               //
               OptixVisibilityMask(255),
               // Flags
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               // SBT
               accIndex, 1, 0,
               // Payload
               currentLevel,
               leafNodeIdOut,
               tMaxOutUInt32);

    // Convert tMax after trace
    float tMaxOut = __uint_as_float(tMaxOutUInt32);

    // We find a hit, now find the voxel level
    // from the distance
    uint32_t globalNodeId = UINT32_MAX;
    uint32_t requiredLevel = svo.LeafDepth();
    if(leafNodeIdOut != UINT32_MAX)
    {
        using SVO = AnisoSVOctreeGPU;
        // We have a hit if code flows here
        // Check if the hit location is valid
        // (cone aperture vs. voxel size check)
        float distance = tMaxOut - ray.tMin;
        float coneDiskDiamSqr = SVO::ConeDiameterSqr(distance, params.pixelAperture);
        float levelVoxelSize = svo.LeafVoxelSize();
        float levelVoxelSizeSqr = levelVoxelSize * levelVoxelSize;
        // Find the level
        float dvRatio = max(0.0f, log2(coneDiskDiamSqr / levelVoxelSizeSqr) * 0.5f);
        requiredLevel = svo.LeafDepth() - static_cast<uint32_t>(floor(dvRatio));
        // Required level is above and found queried level is leaf
        // Ascend to level
        globalNodeId = svo.Ascend(requiredLevel, leafNodeIdOut, svo.LeafDepth());
    }

    // Finally Query the result
    Vector4f locColor = CalcColorSVO(params.renderMode, svo,
                                     ray.ray.getDirection(),
                                     params.pixelAperture,
                                     globalNodeId, requiredLevel);
    // Actual Write
    uint32_t sampleIndex = params.gRayAux[launchIndex].sampleIndex;
    params.gSamples.gValues[sampleIndex] = locColor;
}


__device__ __forceinline__
void KCRadGenSVO()
{
    // TODO:
}

WRAP_FUCTION(__raygen__SVOCamTrace, KCCamTraceSVO);
WRAP_FUCTION(__raygen__SVORadGen, KCRadGenSVO);
WRAP_FUCTION(__miss__SVO, KCMissSVOOptiX);
WRAP_FUCTION(__closesthit__SVO, KCClosestHitSVO);
WRAP_FUCTION(__intersection__SVOMorton32, KCIntersectVoxel<uint32_t>)
WRAP_FUCTION(__intersection__SVOMorton64, KCIntersectVoxel<uint64_t>)