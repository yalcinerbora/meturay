#pragma once

/**

Path Node Related Structures

These structs are used to express chain of paths
which are useful when a tracer needs to hold an entire information
about a path. (i.e MLT BDPT style tracers)

In initial case path node structs are crated to hold "Practical Path Guiding" related
information

*/

#include "RayLib/Vector.h"
#include "RayLib/Constants.h"
#include "RayLib/CoordinateConversion.h"
#include "RayLib/ColorConversion.h"

// Path node is base class for path related data
// It is designed to have minimal memory footprint in order to save
// well memory especially on GPU Hardware
//
// For example only world position is kept
// wi/wo can be generated by hopping the next/previous path nodes and
// calculating these direction using world positions
struct PathNode
{
    using IndexType = uint8_t;
    // No path will self intersect
    static constexpr IndexType InvalidIndex = std::numeric_limits<IndexType>::max();

    // Local Position of the path
    Vector3f                worldPosition;
    // By design all path chains are assumed to have
    // Camera -> Light ordering.
    // Meaning that if you keep accessing next
    // you will end up on the light
    Vector<2, IndexType>    prevNext;

    template <class Node>
    __device__ Vector3f     Wi(const Node* gNodeList, uint32_t pathStartIndex);
    template <class Node>
    __device__ Vector3f     Wo(const Node* gNodeList, uint32_t pathStartIndex);
    template <class Node>
    __device__ Vector3f     NextPos(const Node* gNodeList, uint32_t pathStartIndex);
    template <class Node>
    __device__ Vector3f     PrevPos(const Node* gNodeList, uint32_t pathStartIndex);

    __device__ bool         HasPrev();
    __device__ bool         HasNext();
};

template <class Node>
__device__ inline
Vector3f PathNode::Wi(const Node* gNodeList, uint32_t pathStartIndex)
{
    IndexType next = prevNext[1];
    // Specifically put infinity here to catch some errors
    if(next == InvalidIndex) return Vector3f(INFINITY);
    //
    Vector3f wi = gNodeList[pathStartIndex + next].worldPosition - worldPosition;
    return wi.Normalize();
}

template <class Node>
__device__ inline
Vector3f PathNode::Wo(const Node* gNodeList, uint32_t pathStartIndex)
{
    IndexType prev = prevNext[0];
    // Specifically put infinity here to catch some errors
    if(prev == InvalidIndex) return Vector3f(INFINITY);
    //
    Vector3f wo = gNodeList[pathStartIndex + prev].worldPosition - worldPosition;
    return wo.Normalize();
}

template <class Node>
__device__ inline
Vector3f PathNode::NextPos(const Node* gNodeList, uint32_t pathStartIndex)
{
    IndexType next = prevNext[1];
    // Specifically put infinity here to catch some errors
    if(next == InvalidIndex) return Vector3f(INFINITY);

    return gNodeList[pathStartIndex + next].worldPosition;
}

template <class Node>
__device__ inline
Vector3f PathNode::PrevPos(const Node* gNodeList, uint32_t pathStartIndex)
{
    IndexType prev = prevNext[0];
    // Specifically put infinity here to catch some errors
    if(prev == InvalidIndex) return Vector3f(INFINITY);

    return gNodeList[pathStartIndex + prev].worldPosition;
}

__device__ inline
bool PathNode::HasPrev()
{
    return prevNext[0] != InvalidIndex;
}

__device__ inline
bool PathNode::HasNext()
{
    return prevNext[1] != InvalidIndex;
}

struct PathGuidingNode : public PathNode
{
    Vector3f            radFactor;          // A.k.a path throughput
    Vector3f            totalRadiance;      // Total Radiance

    // Accumulate the generated radiance to the path
    // Radiance provided has to be from the camera (or starting point)
    // of the path so that the class can utilize its own throughput to calculate
    // radiance of that path
    __device__ void     AccumRadiance(const Vector3f& endPointRadiance);

    template <class Node>
    __device__ void     AccumRadianceDownChain(const Vector3f& endPointRadiance,
                                               Node* gLocalChain);
    template <class Node>
    __device__ void     AccumRadianceUpChain(const Vector3f& endPointRadiance,
                                             Node* gLocalChain);
};

struct PPGPathNode : public PathGuidingNode
{
    uint32_t            dataStructIndex;    // Index of an arbitrary data structure
};

struct WFPGPathNode : public PathGuidingNode
{
    Vector2us               packedNormal;
    // For testing
    float                   pdf;

    __device__ Vector3f     Normal() const;
    __device__ void         SetNormal(const Vector3f& normal);
};

__device__ inline
void PathGuidingNode::AccumRadiance(const Vector3f& endPointRadiance)
{
    // Radiance Factor shows the energy ratio between the path start point
    // and this location, so division will give the radiance of that location
    totalRadiance[0] += (radFactor[0] == 0.0f) ? 0.0f : endPointRadiance[0] / radFactor[0];
    totalRadiance[1] += (radFactor[1] == 0.0f) ? 0.0f : endPointRadiance[1] / radFactor[1];
    totalRadiance[2] += (radFactor[2] == 0.0f) ? 0.0f : endPointRadiance[2] / radFactor[2];
}

template <class Node>
__device__ inline
void PathGuidingNode::AccumRadianceDownChain(const Vector3f& endPointRadiance,
                                             Node* gLocalChain)
{
    float luminance = Utility::RGBToLuminance(endPointRadiance);
    //bool largeRadiance = (luminance > 300.0f);
    // Add to yourself
    AccumRadiance(endPointRadiance);

    //if constexpr(std::is_same_v<Node, WFPGPathNode>)
    //{
    //    if(largeRadiance)
    //        printf("l[%f], i[%u], pdf(%f), p(%f, %f, %f)\n",
    //               luminance,
    //               static_cast<uint32_t>(gLocalChain[prevNext[0]].prevNext[1]),
    //               static_cast<WFPGPathNode&>(*this).pdf,
    //               worldPosition[0],
    //               worldPosition[1],
    //               worldPosition[2]);
    //}

    for(IndexType i = prevNext[0];
        i != InvalidIndex;
        i = gLocalChain[i].prevNext[0])
    {
        gLocalChain[i].AccumRadiance(endPointRadiance);

        //if constexpr(std::is_same_v<Node, WFPGPathNode>)
        //{
        //    if(largeRadiance)
        //        printf("l[%f], i[%u], pdf(%f), p(%f, %f, %f)\n",
        //               luminance, static_cast<uint32_t>(i),
        //               static_cast<WFPGPathNode&>(gLocalChain[i]).pdf,
        //               gLocalChain[i].worldPosition[0],
        //               gLocalChain[i].worldPosition[1],
        //               gLocalChain[i].worldPosition[2]);
        //}

    }
}

template <class Node>
__device__ inline
void PathGuidingNode::AccumRadianceUpChain(const Vector3f& endPointRadiance,
                                           Node* gLocalChain)
{
    // Add to yourself
    AccumRadiance(endPointRadiance);
    for(IndexType i = prevNext[1]; i != InvalidIndex;)
    {
        gLocalChain[i].AccumRadiance(endPointRadiance);
        i = gLocalChain[i].prevNext[1];
    }
}

__device__ inline
Vector3f WFPGPathNode::Normal() const
{
    Vector2f normalSphr = Vector2f(static_cast<float>(packedNormal[0]) / 65535.0f,
                                   static_cast<float>(packedNormal[1]) / 65535.0f);
    normalSphr[0] *= MathConstants::Pi * 2.0f;
    normalSphr[0] -= MathConstants::Pi;
    normalSphr[1] *= MathConstants::Pi;
    return Utility::SphericalToCartesianUnit(normalSphr).Normalize();
}

__device__ inline
void WFPGPathNode::SetNormal(const Vector3f& normal)
{
       // Calculate Spherical UV Coordinates of the normal
    Vector2f sphrCoords = Utility::CartesianToSphericalUnit(normal);
    sphrCoords[0] = (sphrCoords[0] + MathConstants::Pi) * MathConstants::InvPi * 0.5f;
    sphrCoords[1] = sphrCoords[1] * MathConstants::InvPi;
    // Due to numerical error this could slightly exceed [0, 65535]
    // clamp it
    Vector2i sphrUnormInt = Vector2i(static_cast<int32_t>(sphrCoords[0] * 65535.0f),
                                     static_cast<int32_t>(sphrCoords[1] * 65535.0f));
    sphrUnormInt.ClampSelf(0, 65535);
    packedNormal = Vector2us(sphrUnormInt[0], sphrUnormInt[1]);
}

__global__
static void KCInitializePPGPaths(PPGPathNode* gPathNodes,
                                 uint32_t totalNodeCount)
{
    uint32_t globalId = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalId < totalNodeCount)
    {
        PPGPathNode node;
        node.dataStructIndex = UINT32_MAX;
        node.radFactor = Vector3f(1.0f);
        node.prevNext = Vector<2, PathNode::IndexType>(PathNode::InvalidIndex);
        node.totalRadiance = Zero3;
        node.worldPosition = Zero3;

        gPathNodes[globalId] = node;
    }
}

__global__
static void KCInitializePGPaths(PathGuidingNode* gPathNodes,
                                 uint32_t totalNodeCount)
{
    uint32_t globalId = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalId < totalNodeCount)
    {
        PathGuidingNode node;
        node.radFactor = Vector3f(1.0f);
        node.prevNext = Vector<2, PathNode::IndexType>(PathNode::InvalidIndex);
        node.totalRadiance = Zero3;
        node.worldPosition = Zero3;

        gPathNodes[globalId] = node;
    }
}

__global__
static void KCInitializeWFPGPaths(WFPGPathNode* gPathNodes,
                                  uint32_t totalNodeCount)
{
    uint32_t globalId = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalId < totalNodeCount)
    {
        WFPGPathNode node;
        node.radFactor = Vector3f(1.0f);
        node.prevNext = Vector<2, PathNode::IndexType>(PathNode::InvalidIndex);
        node.totalRadiance = Zero3;
        node.worldPosition = Zero3;

        gPathNodes[globalId] = node;
    }
}