#include "AnisoSVO.cuh"
#include "RayLib/ColorConversion.h"

#include <cub/cub.cuh>

template <uint32_t THREAD_PER_BLOCK>
__global__ __launch_bounds__(THREAD_PER_BLOCK)
void KCReduceVoxelPayload(// I-O
                          AnisoSVOctreeGPU treeGPU,
                          // Input
                          const uint32_t* gVoxelIndexOffsets,
                          const uint64_t* gUniqueVoxels,
                          // non-unique voxel index array
                          const uint32_t* gSortedVoxelIndices,
                          // Voxel payload that will be reduced
                          const HitKey* gVoxelLightKeys,
                          const Vector2us* gVoxelNormals,
                          // Binary Search for light
                          const HitKey* gLightKeys,
                          const GPULightI** gLights,
                          uint32_t lightCount,
                          // Constants
                          uint32_t uniqueVoxCount,
                          uint32_t lightKeyCount,
                          const AABB3f svoAABB,
                          uint32_t resolutionXYZ)
{
    auto UnpackNormal = [](const Vector2us& packedNormal) -> Vector3f
    {
        Vector2f normalSphr = Vector2f(static_cast<float>(packedNormal[0]) / 65535.0f,
                                       static_cast<float>(packedNormal[1]) / 65535.0f);
        normalSphr[0] *= MathConstants::Pi * 2.0f;
        normalSphr[0] -= MathConstants::Pi;
        normalSphr[1] *= MathConstants::Pi;
        return Utility::SphericalToCartesianUnit(normalSphr).Normalize();
    };

    // Constants
    static constexpr uint32_t DATA_PER_THREAD = 2;
    static constexpr uint32_t MEAN_COUNT = 2;
    static constexpr uint32_t WARP_PER_BLOCK = THREAD_PER_BLOCK / WARP_SIZE;

    // Cub reduction broadcast, sort etc.
    using WarpNormalReduce = cub::WarpReduce<Vector3f>;
    using WarpUIntReduce = cub::WarpReduce<uint32_t>;
    using WarpLumReduce = cub::WarpReduce<Vector2f>;
    __shared__ union
    {
        typename WarpNormalReduce::TempStorage sNormalReduceMem[WARP_PER_BLOCK];
        typename WarpUIntReduce::TempStorage sUIntReduceMem[WARP_PER_BLOCK];
        typename WarpLumReduce::TempStorage sLumReduceMem[WARP_PER_BLOCK];
    } sMem;

    // Grid Stride parameters:
    // This kernel should be called with grid stride parameters
    // (enough threads to saturate the GPU)
    // each warp will work on a single unique voxel
    // and reduce the normals (using k-mean clustering) and light emittance
    // Also, gridSize should be multiple of warp size (this should always be true
    // because of occupancy)
    const uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t warpId = threadId / WARP_SIZE;
    const uint32_t gridSize = blockDim.x * gridDim.x;
    const uint32_t totalWarpCount = gridSize / WARP_SIZE;
    const uint32_t warpLocalId = (threadId % WARP_SIZE);
    const uint32_t blockLocalWarpId = warpId % (blockDim.x / WARP_SIZE);
    const bool isWarpLeader = (warpLocalId == 0);

    // Grid Stride Loop: A warp (32 threads) for each voxel
    for(uint32_t voxelIndex = warpId; voxelIndex < uniqueVoxCount;
        voxelIndex += totalWarpCount)
    {
        // Normals (which will undergo clustering)
        Vector3f normals[DATA_PER_THREAD];
        // Constants (will be fetched and broadcasted by the warp leader)
        uint64_t mortonCode;
        Vector2ui reduceRange;

        // Load Voxel Key, offset etc.
        // TODO: you can do coalesced fetch here but we may implement it later
        if(isWarpLeader)
        {
            mortonCode = gUniqueVoxels[voxelIndex];
            reduceRange = Vector2ui(gVoxelIndexOffsets[voxelIndex],
                                    gVoxelIndexOffsets[voxelIndex + 1]);
        }

        // Use warp functions to broadcast
        mortonCode = cub::ShuffleIndex<WARP_SIZE>(mortonCode, 0, 0xFFFFFFFF);
        reduceRange = cub::ShuffleIndex<WARP_SIZE>(reduceRange, 0, 0xFFFFFFFF);

        // Convenience duplicateVoxelCount
        uint32_t dupVoxCount = reduceRange[1] - reduceRange[0];

        // Each warp thread convert morton code to world position
        // we will use it to query SVO
        Vector3ui denseIndex = MortonCode::Decompose<uint64_t>(mortonCode);
        Vector3f worldPos = treeGPU.VoxelToWorld(denseIndex);

        /////////////////////////////////////
        //    K-MEANS NORMAL CLUSTERING    //
        /////////////////////////////////////
        // Start clustering, we need to assume normals may not fit into
        // register memory, we need to iterate over multiple times maybe
        // Determine it
        static constexpr uint32_t TOTAL_REGISTER_SPACE = DATA_PER_THREAD * WARP_SIZE;
        const uint32_t passAmount = (dupVoxCount + TOTAL_REGISTER_SPACE - 1) / TOTAL_REGISTER_SPACE;
        uint32_t currentPass = 0;

        // Before starting the clustering load the normals
        auto LoadNormals = [&](uint32_t passId)
        {
            // Load from passId
            #pragma unroll
            for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
            {
                uint32_t passOffset = passId * TOTAL_REGISTER_SPACE;
                uint32_t localOffset = i * WARP_SIZE + warpLocalId;
                uint32_t combinedOffset = passOffset + localOffset;
                // Only fetch if in range
                if(combinedOffset < dupVoxCount)
                {
                    uint32_t voxelIndex = gSortedVoxelIndices[reduceRange[0] + combinedOffset];
                    Vector2us packedNormal = gVoxelNormals[voxelIndex];
                    normals[i] = UnpackNormal(packedNormal);
                }
                else normals[i] = Zero3f;
            }
        };

        // Initially Load first pass of normals
        LoadNormals(currentPass);

        // Broadcast the very first normal to other
        // threads as an initial mean
        // Double buffer here
        Vector3f meanNormals[2][MEAN_COUNT];
        uint32_t counts[2][MEAN_COUNT];

        // Select two extremes as means
        meanNormals[0][0] = cub::ShuffleIndex<WARP_SIZE>(normals[0], 0, 0xFFFFFFFF);
        meanNormals[0][1] = -meanNormals[0][0];
        // Init accum counts
        counts[1][0] = 0;
        counts[1][1] = 0;

        // Start cluster iteration
        // Do 2-means clustering
        static constexpr uint32_t K_MEANS_CLUSTER_ITER_COUNT = 4;
        #pragma unroll
        for(uint32_t kMeanPassId = 0; kMeanPassId < K_MEANS_CLUSTER_ITER_COUNT; kMeanPassId++)
        {
            // Normalize the means (at the end of the last iteration
            // we don't do that since we need the unnormalized values to create distribution etc
            meanNormals[0][0].NormalizeSelf();
            meanNormals[0][1].NormalizeSelf();

            // Cluster the data using the means
            // Do this pass by pass fashion
            // (we may not have enough space to load every normal)
            for(int pass = 0; pass < passAmount; pass++)
            {
                // Load the normals into register space
                // Skip for the first pass since we already have some normals
                if(pass != 0)
                {
                    uint32_t passId = currentPass % passAmount;
                    LoadNormals(passId);
                }

                #pragma unroll
                for(uint32_t nIndex = 0; nIndex < DATA_PER_THREAD; nIndex++)
                {
                    // Skip if we are on an edge case
                    if(normals[nIndex] == Vector3f(0.0f)) continue;

                    // Do the mean calculation
                    // Calculate angular distance and choose your mean
                    float angularDist0 = 1.0f - meanNormals[0][0].Dot(normals[nIndex]);
                    float angularDist1 = 1.0f - meanNormals[0][1].Dot(normals[nIndex]);

                    uint32_t updateIndex = (angularDist0 >= angularDist1) ? 0 : 1;
                    meanNormals[1][updateIndex] += normals[nIndex];
                    counts[1][updateIndex] += 1;
                }

                // We processed this pass adjust the currentPass
                currentPass++;
            }
            // Prevent unnecessary loading by not incrementing the last pass;
            // Assume there are total of 3 passes (4 normal per thread, 32 thread in a warp)
            // There are [257 - 384] normals
            // i.e.
            //  K-Means Iter0: [0, 1, 2]
            //  K-Means Iter1: [2, 0, 1]
            //  K-Means Iter2: [1, 2, 0] etc.
            // We prevent a single bulk load with this style
            currentPass--;

            // Now get ready for the next iteration
            // "meanNormals", "counts"  variables are on local space
            // Reduce and broadcast it to all threads
            meanNormals[0][0] = WarpNormalReduce(sMem.sNormalReduceMem[blockLocalWarpId]).Sum(meanNormals[1][0]);
            meanNormals[0][1] = WarpNormalReduce(sMem.sNormalReduceMem[blockLocalWarpId]).Sum(meanNormals[1][1]);
            counts[0][0] = WarpUIntReduce(sMem.sUIntReduceMem[blockLocalWarpId]).Sum(counts[1][0]);
            counts[0][1] = WarpUIntReduce(sMem.sUIntReduceMem[blockLocalWarpId]).Sum(counts[1][1]);
            // TODO: sync is required here? probably not (at least for SM 3.0 or newer)

            // Clean the accumulation buffers
            meanNormals[1][0] = Zero3f;
            meanNormals[1][1] = Zero3f;
            counts[1][0] = 0;
            counts[1][1] = 0;

            // Broadcast the new means to threads
            meanNormals[0][0] = cub::ShuffleIndex<WARP_SIZE>(meanNormals[0][0], 0, 0xFFFFFFFF);
            meanNormals[0][1] = cub::ShuffleIndex<WARP_SIZE>(meanNormals[0][1], 0, 0xFFFFFFFF);
        }

        // Iteration is complete, choose the cluster that has maximum amount of normals in it
        uint32_t normalIndex = (counts[0][0] >= counts[0][1]) ? 0 : 1;
        if(isWarpLeader)
        {
            meanNormals[0][normalIndex] *= (1.0f / static_cast<float>(counts[0][normalIndex]));
            uint32_t normalLength = meanNormals[0][normalIndex].Length();
        }
        // Do a final broadcast (normals will be used to calculate emittance)
        meanNormals[0][0] = cub::ShuffleIndex<WARP_SIZE>(meanNormals[0][normalIndex], 0, 0xFFFFFFFF);
        ///////////////////////////////////////
        // K-MEANS NORMAL CLUSTERING IS DONE //
        ///////////////////////////////////////

        ////////////////////////////////////////
        //  NOW LIGHT EMITTANCE ACCUMULATION  //
        ////////////////////////////////////////
        // Then calculate the light luminance for each non unique voxel
        // For each colliding voxel
        Vector2f combinedLuminance = Zero2f;

        HitKey lightKeys[DATA_PER_THREAD];
        for(int pass = 0; pass < passAmount; pass++)
        {
            // Load keys
            #pragma unroll
            for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
            {
                uint32_t passOffset = pass * TOTAL_REGISTER_SPACE;
                uint32_t localOffset = i * WARP_SIZE + warpLocalId;
                uint32_t combinedOffset = passOffset + localOffset;
                // Only fetch if in range
                if(combinedOffset < dupVoxCount)
                {
                    uint32_t voxelIndex = gSortedVoxelIndices[reduceRange[0] + combinedOffset];
                    lightKeys[i] = gVoxelLightKeys[voxelIndex];
                }
                else lightKeys[i] = HitKey::InvalidKey;
            }

            // TODO: Unless voxel size is extremely large, these hit keys are probably the same
            // so do a sort here, and then do an adjacent difference to prevent duplicate work
            // Now query the light directly

            Vector3f normalizedNormal = meanNormals[0][0].NormalizeSelf();

            // Now we can do the light search etc.
            for(uint32_t i = 0; i < DATA_PER_THREAD; i++)
            {
                HitKey lightKey = lightKeys[i];
                if(lightKey == HitKey::InvalidKey) continue;
                // Binary search the light with key
                float lightIndexF;
                bool found = GPUFunctions::BinarySearchInBetween(lightIndexF, lightKey,
                                                                 gLightKeys, lightCount);
                uint32_t lightIndex = static_cast<uint32_t>(lightIndexF);
                assert(found);
                if(!found)
                {
                    KERNEL_DEBUG_LOG("Error: SVO light not found!\n");
                    continue;
                }
                const GPULightI* gLight = gLights[lightIndex];

                // Query both sides of the surface
                // Towards normal
                Vector3f radiance = gLight->Emit(normalizedNormal, worldPos, UVSurface{});
                combinedLuminance[0] += Utility::RGBToLuminance(radiance);
                // Query both sides of the surface
                radiance = gLight->Emit(-normalizedNormal, worldPos, UVSurface{});
                combinedLuminance[1] += Utility::RGBToLuminance(radiance);
            }
        }

        // Reduce the combined luminance on the leader thread
        combinedLuminance = WarpLumReduce(sMem.sLumReduceMem[blockLocalWarpId]).Sum(combinedLuminance);
        if(isWarpLeader)
        {
            // Only Set Luminance if this voxel contains a light surface
            if(combinedLuminance != Vector3f(0.0f))
            {
                // Don't forget to average
                combinedLuminance /= static_cast<float>(dupVoxCount);
                // We are setting initial sample count to this voxel
                // there shouldn't be any updates to this voxel anyway but just to be sure
                Vector2ui initialSampleCount = Vector2ui(100);
                combinedLuminance *= Vector2f(initialSampleCount);
                // Set the combined values
                treeGPU.SetLeafRadiance(mortonCode, combinedLuminance, initialSampleCount);
            }

            if(meanNormals[0][0].HasNaN())
            {
                printf("NaN Normal is Generated!\n");
            }
            treeGPU.SetLeafNormal(mortonCode, meanNormals[0][0]);
        }
        // All Done!
    }
}