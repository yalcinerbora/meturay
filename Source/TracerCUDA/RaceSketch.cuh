#pragma once

#include "L2HashGroup.cuh"
#include "SRPHashGroup.cuh"

#include "RayLib/AABB.h"
#include "RayLib/Constants.h"

struct PathGuidingNode;
class CudaSystem;

class RaceSketchGPU
{
    private:
        float*                  gSketch;
        double*                 gTotalCount;

        //AngleGroupCPU<5>   hashesCPU;
        L2HashGroupGPU<5>       hashes;
        //SRPHashGroupGPU<5>      hashes;
        uint32_t                numHash;
        uint32_t                numPartition;
        AABB3f                  sceneExtents;


        friend class            RaceSketchCPU;

        __device__
        static void             CopyData(float(&data)[5],
                                         const Vector3f& worldPos,
                                         const Vector2f& direction);

    protected:
    public:
        __device__ void         AtomicAddData(const Vector3f& worldPos,
                                              const Vector2f& direction,
                                              float value);

        __device__ float        Probability(const Vector3f& worldPos,
                                            const Vector2f& direction) const;

};

class RaceSketchCPU
{
    private:
        DeviceMemory        mem;
        //AngleGroupCPU<5>   hashesCPU;
        L2HashGroupCPU<5>   hashesCPU;
        //SRPHashGroupCPU<5>  hashesCPU;
        RaceSketchGPU       sketchGPU;

    protected:
    public:
        // Constructors & Destructor
                        RaceSketchCPU(uint32_t numHash,
                                      uint32_t numPartition,
                                      float bucketWidth,
                                      uint32_t seed);

        void            SetSceneExtent(const AABB3f& sceneExtents);
        void            HashRadianceAsPhotonDensity(const PathGuidingNode* dPGNodes,
                                                    uint32_t totalNodeCount,
                                                    uint32_t maxPathNodePerRay,
                                                    const CudaSystem& system);

        void            GetSketchToCPU(std::vector<float>& sketchList,
                                       uint64_t& totalSamples) const;


        RaceSketchGPU   SketchGPU() const;
        uint32_t        HashCount() const;
        uint32_t        PartitionCount() const;

};

__device__ inline
void RaceSketchGPU::CopyData(float(&data)[5],
                             const Vector3f& worldPos,
                             const Vector2f& direction)
{
    #pragma unroll
    for(int i = 0; i < 3; i++)
        data[i] = worldPos[i];
    #pragma unroll
    for(int i = 0; i < 2; i++)
        data[i + 3] = direction[i];
}

__device__ inline
void RaceSketchGPU::AtomicAddData(const Vector3f& worldPos,
                                  const Vector2f& direction,
                                  float value)
{
    // Flatten the direction and world position vectors
    float dimData[5];
    CopyData(dimData, worldPos, direction);

    // Normalize the Data to fit a range [0, 1]
    Vector3f sceneRange = sceneExtents.Span();
    dimData[0] = (dimData[0] - sceneExtents.Min()[0]) / sceneRange[0];
    dimData[1] = (dimData[1] - sceneExtents.Min()[1]) / sceneRange[1];
    dimData[2] = (dimData[2] - sceneExtents.Min()[2]) / sceneRange[2];
    // Spherical coords
    dimData[3] = dimData[3] * MathConstants::InvPiHalf + 0.5f;
    dimData[4] = dimData[4] * MathConstants::InvPi;
    //// Tentify the theta
    //dimData[3] = 2.0f * fabs(dimData[3] - 0.5f);

    //float lengthSqr = (dimData[0] * dimData[0] +
    //                   dimData[1] * dimData[1] +
    //                   dimData[2] * dimData[2] +
    //                   dimData[3] * dimData[3] +
    //                   dimData[4] * dimData[4]);
    //float l = (1.0f / sqrtf(lengthSqr));
    //dimData[0] *= l;
    //dimData[1] *= l;
    //dimData[2] *= l;
    //dimData[3] *= l;
    //dimData[4] *= l;

    // Quantize the radiance to make it value
    // TODO: change this to proper radiance->photon calculation
    float val = value;

    bool hasOverflow = false;

    // Write to the sketch
    for(uint32_t i = 0; i < numHash; i++)
    {
        uint32_t hashOut;
        hashes.Hash(hashOut, dimData, i);

        if(hashOut >= numPartition)
            hasOverflow = true;

        // Wrap the function for negative indices
        hashOut %= numPartition;

        atomicAdd(gSketch + (i * numPartition + hashOut), val);
    }

    if(hasOverflow) printf("Has oflow\n");

    atomicAdd(gTotalCount, val);
}

__device__ inline
float RaceSketchGPU::Probability(const Vector3f& worldPos,
                                 const Vector2f& direction) const
{
    // Flatten the direction and world position vectors
    float dimData[5];
    CopyData(dimData, worldPos, direction);

    // Normalize the Data to fit a range [0, 1]
    Vector3f sceneRange = sceneExtents.Span();
    dimData[0] = (dimData[0] - sceneExtents.Min()[0]) / sceneRange[0];
    dimData[1] = (dimData[1] - sceneExtents.Min()[1]) / sceneRange[1];
    dimData[2] = (dimData[2] - sceneExtents.Min()[2]) / sceneRange[2];
    // Spherical coords
    dimData[3] = dimData[3] * MathConstants::InvPiHalf + 0.5f;
    dimData[4] = dimData[4] * MathConstants::InvPi;
    //// Tentify the theta
    //dimData[3] = 2.0f * fabs(dimData[3] - 0.5f);
    //float lengthSqr = (dimData[0] * dimData[0] +
    //                   dimData[1] * dimData[1] +
    //                   dimData[2] * dimData[2] +
    //                   dimData[3] * dimData[3] +
    //                   dimData[4] * dimData[4]);
    //float l = (1.0f / sqrtf(lengthSqr));
    //dimData[0] *= l;
    //dimData[1] *= l;
    //dimData[2] *= l;
    //dimData[3] *= l;
    //dimData[4] *= l;



    double total = 1.0;// *gTotalCount;
    double totRecip = 1.0 / total;
    double nHRecip = 1.0 / static_cast<double>(numHash);
    double val = 0;
    for(uint32_t i = 0; i < numHash; i++)
    {
        uint32_t hashOut;
        hashes.Hash(hashOut, dimData, i);

        // Wrap the function for "negative" indices
        hashOut %= numPartition;

        double sketchVal = static_cast<double>(gSketch[i * numPartition + hashOut]);
        val += sketchVal * nHRecip * totRecip;
    }
    return val;

    // TODO: reason about the proper arithmetic here
    // should we use double or any other way (currently float)
    //return static_cast<double>(val) / static_cast<double>(total);
}

inline void RaceSketchCPU::SetSceneExtent(const AABB3f& sceneExtents)
{
    sketchGPU.sceneExtents = sceneExtents;
}

inline RaceSketchGPU RaceSketchCPU::SketchGPU() const
{
    return sketchGPU;
}

inline uint32_t RaceSketchCPU::HashCount() const
{
    return sketchGPU.numHash;
}

inline uint32_t RaceSketchCPU::PartitionCount() const
{
    return sketchGPU.numPartition;
}