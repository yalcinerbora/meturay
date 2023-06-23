#include  "OctreeOptiXPTX.cuh"
#include "RayLib/RandomColor.h"
#include "WFPGTracerKC.cuh"

#include <optix_device.h>

extern "C" __constant__ OctreeAccelParams params;

// Meta Closest Hit Shader
__device__ __forceinline__
void KCClosestHitSVO()
{
    const unsigned int leafId = optixGetPrimitiveIndex();
    optixSetPayload_1(leafId);
    optixSetPayload_2(__float_as_uint(optixGetRayTmax()));
}

__device__ __forceinline__
void KCBaseAcceleratorClosestHit()
{
    // SBT is kinda hacky here we cannot acquire payload
    // Fortunately we dont have to
    // TODO: Design this better, if it will stay
    float3 rayOrig = optixGetWorldRayOrigin();
    float3 rayDir = optixGetWorldRayDirection();
    float tMax = optixGetRayTmax();

    RayF ray = RayF(Vector3f(rayDir.x, rayDir.y, rayDir.z),
                    Vector3f(rayOrig.x, rayOrig.y, rayOrig.z));
    // Estimate Hit Position
    Vector3f worldPos = ray.AdvancedPos(tMax + MathConstants::Epsilon);
    uint32_t leafId;
    bool found = params.svo.NearestNodeIndex(leafId, worldPos, params.svo.LeafDepth(), true);

    if(!found) printf("NotFound!, tMax %f, leaf: %u, p:(%f, %f, %f)\n",
                      tMax, leafId,
                      worldPos[0], worldPos[1], worldPos[2]);

    optixSetPayload_1(leafId);
    optixSetPayload_2(__float_as_uint(tMax));
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
    RayReg ray = RayReg(params.ct.gRays, launchIndex);
    // Actual SVO structure will be used to expand the hit location wrt.
    // query level
    const AnisoSVOctreeGPU& svo = params.svo;

    // Start level is leaf or the offseted leaf
    // (query offset is used  for debugging only)
    int startLevel = static_cast<int>(svo.LeafDepth() - params.ct.maxQueryOffset);
    startLevel = max(1, startLevel);
    int accIndex = (startLevel - 1);
    // Ray Payload
    // In
    uint32_t currentLevel = static_cast<int>(startLevel);
    // Out
    uint32_t leafNodeIdOut = UINT32_MAX;
    uint32_t tMaxOutUInt32 = __float_as_uint(ray.tMax);


    OptixTraversableHandle traversable = (params.utilizeSceneAccelerator)
                                            ? params.sceneBVH
                                            : params.octreeLevelBVHs[accIndex];
    uint32_t sbtOffset = (params.utilizeSceneAccelerator)
                            ? 0u
                            : static_cast<uint32_t>(accIndex);

    // Trace Call
    optixTrace(// Accelerator
               traversable,
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
               sbtOffset, 1, 0,
               // Payload
               currentLevel,
               leafNodeIdOut,
               tMaxOutUInt32);
    //optixTrace(// Accelerator
    //           params.octreeLevelBVHs[accIndex],
    //           // Ray Input
    //           make_float3(ray.ray.getPosition()[0],
    //                       ray.ray.getPosition()[1],
    //                       ray.ray.getPosition()[2]),
    //           make_float3(ray.ray.getDirection()[0],
    //                       ray.ray.getDirection()[1],
    //                       ray.ray.getDirection()[2]),
    //           ray.tMin,
    //           ray.tMax,
    //           0.0f,
    //           //
    //           OptixVisibilityMask(255),
    //           // Flags
    //           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    //           // SBT
    //           accIndex, 1, 0,
    //           // Payload
    //           currentLevel,
    //           leafNodeIdOut,
    //           tMaxOutUInt32);

    // Convert tMax after trace
    float tMaxOut = __uint_as_float(tMaxOutUInt32);

    // For scene accelerator we will find leaf
    // Change currentLevel to leaf depth
    currentLevel = (params.utilizeSceneAccelerator)
                    ? svo.LeafDepth()
                    : currentLevel;

    // We find a hit, now find the voxel level
    // from the distance
    uint32_t globalNodeId = UINT32_MAX;
    uint32_t requiredLevel = currentLevel;
    if(leafNodeIdOut != UINT32_MAX)
    {
        using SVO = AnisoSVOctreeGPU;
        // We have a hit if code flows here
        // Check if the hit location is valid
        // (cone aperture vs. voxel size check)
        float distance = tMaxOut - ray.tMin;
        float coneDiskDiamSqr = SVO::ConeDiameterSqr(distance, params.pixelOrConeAperture);
        float levelVoxelSize = svo.LevelVoxelSize(currentLevel);
        float levelVoxelSizeSqr = levelVoxelSize * levelVoxelSize;
        // Find the level
        float dvRatio = max(0.0f, log2(coneDiskDiamSqr / levelVoxelSizeSqr) * 0.5f);
        requiredLevel = currentLevel - static_cast<uint32_t>(floor(dvRatio));

        // Clamp the required level according to the "params.maxQueryOffset"
        requiredLevel = min(requiredLevel, svo.LeafDepth() - params.ct.maxQueryOffset);
        // Required level is above and found queried level is leaf
        // Ascend to level
        globalNodeId = svo.Ascend(requiredLevel, leafNodeIdOut, currentLevel);
    }

    // Finally Query the result
    Vector3f hitPos = ray.ray.AdvancedPos(tMaxOut);
    Vector4f locColor = CalcColorSVO(params.ct.renderMode, svo,
                                     ray.ray.getDirection(),
                                     hitPos,
                                     params.pixelOrConeAperture,
                                     globalNodeId, requiredLevel);
    // Actual Write
    uint32_t sampleIndex = params.ct.gRayAux[launchIndex].sampleIndex;
    params.ct.gSamples.gValues[sampleIndex] = locColor;
}

__device__ __forceinline__
void KCRadGenSVO()
{
    auto ProjectionFunc = [](const Vector2i& localPixelId,
                             const Vector2i& segmentSize,
                             Vector2f xi)
    {
        Vector2f st = Vector2f(localPixelId) + xi;
        st /= Vector2f(segmentSize);
        st = Utility::CocentricOctohedralWrap(st);
        Vector3f result = Utility::CocentricOctohedralToDirection(st);
        Vector3f dirYUp = Vector3(result[1], result[2], result[0]);
        return dirYUp;
    };

    // Sanity Check (for payload)
    static_assert(sizeof(uint32_t) == sizeof(float));
    // We Launch linearly
    const uint32_t launchDim = optixGetLaunchDimensions().x;
    const uint32_t launchIndex = optixGetLaunchIndex().x;
    //
    const Vector2i fieldDim = params.rg.fieldSegments.FieldDim();
    int32_t threadPerField = fieldDim.Multiply();
    const uint32_t fieldWriteIndex = launchIndex / threadPerField;
    const uint32_t binIndex = fieldWriteIndex + params.rg.binOffset;
    const uint32_t binThreadIndex = launchIndex % threadPerField;
    // Should we check this ??
    if(launchIndex >= launchDim) return;
    // Generate the ray using field information
    Vector4f originAndTmin = params.rg.dRadianceFieldRayOrigins[binIndex];
    Vector3f rayOrigin = Vector3f(originAndTmin);
    Vector2f tMinMax = Vector2f(originAndTmin[3], FLT_MAX);

    // Determine block validity using tMin (which should be NAN)
    // TODO: Ask on the OptiX forums
    // isnan() does not work
    //if(isnan(tMinMax[0])) return;
    if(tMinMax[0] != tMinMax[0]) return;

    Vector2i rayId(binThreadIndex % fieldDim[0],
                   binThreadIndex / fieldDim[0]);
    // Project using co-centric octohedral projection (mapping)
    // Globally jitter the field (entire field is offseted by this
    // value)
    Vector2f projJitter = params.rg.dProjJitters[binIndex];
    Vector3f rayDir = ProjectionFunc(rayId, fieldDim, projJitter);

    //if(fieldWriteIndex == 1)
    //{
    //    printf("[%u] D(%f, %f, %f) Length(%f)\n",
    //           binThreadIndex, rayDir[0], rayDir[1], rayDir[2],
    //           rayDir.Length());
    //}

    // Calculate tMax from SVO AABB
    // Actual SVO structure will be used to expand the hit location wrt.
    // query level
    // This would work only if the ray is inside the SVO AABB
    // for path guiding all rays will be launched inside the scene
    // so this should be OK.
    // (Unlike camera rays may originate from outside of the scene)
    const AnisoSVOctreeGPU& svo = params.svo;
    Vector2f tMMOut;
    RayF(rayDir, rayOrigin).IntersectsAABB(tMMOut,
                                           svo.OctreeAABB().Min(),
                                           svo.OctreeAABB().Max(),
                                           tMinMax);
    // Epsilon this out for good measure
    tMinMax[1] = tMMOut[1] + MathConstants::VeryLargeEpsilon;

    // Start level is leaf or the offseted leaf
    // (query offset is used  for debugging only)
    int startLevel = static_cast<int>(svo.LeafDepth());
    startLevel = max(1, startLevel);
    int accIndex = (startLevel - 1);
    // Ray Payload
    // In
    uint32_t currentLevel = svo.LeafDepth();
    // Out
    uint32_t leafNodeIdOut = UINT32_MAX;
    uint32_t tMaxOutUInt32 = __float_as_uint(tMinMax[1]);

    OptixTraversableHandle traversable = (params.utilizeSceneAccelerator)
                                            ? params.sceneBVH
                                            : params.octreeLevelBVHs[accIndex];
    uint32_t sbtOffset = (params.utilizeSceneAccelerator)
                            ? 0u
                            : static_cast<uint32_t>(accIndex);

    // Trace Call
    optixTrace(// Accelerator
               traversable,
               // Ray Input
               make_float3(rayOrigin[0], rayOrigin[1], rayOrigin[2]),
               make_float3(rayDir[0], rayDir[1], rayDir[2]),
               tMinMax[0], tMinMax[1],
               0.0f,
               //
               OptixVisibilityMask(255),
               // Flags
               OPTIX_RAY_FLAG_DISABLE_ANYHIT |
               OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               // SBT
               sbtOffset, 1, 0,
               // Payload
               currentLevel,
               leafNodeIdOut,
               tMaxOutUInt32);

    // Convert tMax after trace
    float tMaxOut = __uint_as_float(tMaxOutUInt32);

    // We find a hit, now find the voxel level
    // from the distance
    bool isLeaf = false;
    uint32_t globalNodeId = UINT32_MAX;
    uint32_t requiredLevel = svo.LeafDepth();
    if(leafNodeIdOut != UINT32_MAX)
    {
        using SVO = AnisoSVOctreeGPU;
        // We have a hit if code flows here
        // Check if the hit location is valid
        // (cone aperture vs. voxel size check)
        float distance = tMaxOut - tMinMax[0];
        float coneDiskDiamSqr = SVO::ConeDiameterSqr(distance, params.pixelOrConeAperture);
        float levelVoxelSize = svo.LeafVoxelSize();
        float levelVoxelSizeSqr = levelVoxelSize * levelVoxelSize;
        // Find the level
        float dvRatio = max(0.0f, log2(coneDiskDiamSqr / levelVoxelSizeSqr) * 0.5f);
        requiredLevel = svo.LeafDepth() - static_cast<uint32_t>(floor(dvRatio));
        // Required level is above and found queried level is leaf
        // Ascend to level
        globalNodeId = svo.Ascend(requiredLevel, leafNodeIdOut, svo.LeafDepth());
        isLeaf = (globalNodeId == leafNodeIdOut);
    }

    // Interpolated
    //Vector3f hitPos = rayOrigin + rayDir * tMaxOut;
    //float radiance = ReadInterpolatedRadiance(hitPos, rayDir,
    //                                          params.pixelOrConeAperture,
    //                                          requiredLevel, svo);

    // Nearest
    float radiance = svo.ReadRadiance(rayDir, params.pixelOrConeAperture,
                                      globalNodeId, isLeaf);
    //float radiance = svo.ReadRadiance(rayDir, params.pixelOrConeAperture,
    //                                  leafNodeIdOut, true);
    if(radiance == 0.0f)
        radiance = MathConstants::VeryLargeEpsilon;

    // Now write
    float* dataRange = params.rg.fieldSegments.FieldRadianceArray(fieldWriteIndex);
    dataRange[binThreadIndex] = radiance;

    float* distRange = params.rg.fieldSegments.FieldDistanceArray(fieldWriteIndex);
    distRange[binThreadIndex] = tMaxOut;
}

WRAP_FUCTION(__raygen__SVOCamTrace, KCCamTraceSVO);
WRAP_FUCTION(__raygen__SVORadGen, KCRadGenSVO);
WRAP_FUCTION(__miss__SVO, KCMissSVOOptiX);
WRAP_FUCTION(__closesthit__SVO, KCClosestHitSVO);
WRAP_FUCTION(__closesthit__Scene, KCBaseAcceleratorClosestHit);
WRAP_FUCTION(__intersection__SVOMorton32, KCIntersectVoxel<uint32_t>)
WRAP_FUCTION(__intersection__SVOMorton64, KCIntersectVoxel<uint64_t>)