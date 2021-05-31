#pragma once

/**

Path Node Related Structures

These structs are used to express chain of paths
which are usefull when a tracer needs to hold an entire information
about a path. (i.e MLT BDPT style tracers)

In initial case path node structs are crated to hold "Practical Path Guiding" related
information

*/

#include "RayLib/Vector.h"

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
           
    __device__ bool         HasPrev();
    __device__ bool         HasNext();
};

template <class Node>
__device__ __forceinline__ 
Vector3f PathNode::Wi(const Node* gNodeList, uint32_t pathStartIndex)
{
    IndexType next = prevNext[1];
    // Specifically put infinty here to catch some errors
    if(next == InvalidIndex) return Vector3f(INFINITY);
    //
    Vector3f wi = gNodeList[pathStartIndex + next].worldPosition - worldPosition;
    return wi.Normalize();
}

template <class Node>
__device__ __forceinline__ 
Vector3f PathNode::Wo(const Node* gNodeList, uint32_t pathStartIndex)
{
    IndexType prev = prevNext[0];
    // Specifically put infinty here to catch some errors
    if(prev == InvalidIndex) return Vector3f(INFINITY);
    //
    Vector3f wi = gNodeList[pathStartIndex + prev].worldPosition - worldPosition;
    return wi.Normalize();
}

__device__ __forceinline__ 
bool PathNode::HasPrev()
{
    return prevNext[0] != InvalidIndex;
}

__device__ __forceinline__ 
bool PathNode::HasNext()
{
    return prevNext[1] != InvalidIndex;
}

struct PathGuidingNode : public PathNode
{
    Vector3f            radFactor;          // A.k.a path throughput
    uint32_t            nearestDTreeIndex;
    Vector3f            totalRadiance;      // Total Radiance
    
    // Accumulate the generated radiance to the path
    // Radiance provided has to be from the camera (or starting point)
    // of the path so that the class can utilize its own throuhput to calculate
    // radiance of that path
    __device__ void     AccumRadiance(const Vector3f& camRadiance);
    __device__ void     AccumRadianceDownChain(const Vector3f& camRadiance,
                                               PathGuidingNode* gLocalChain);
    __device__ void     AccumRadianceUpChain(const Vector3f& camRadiance,
                                             PathGuidingNode* gLocalChain);
};

__device__ __forceinline__
void PathGuidingNode::AccumRadiance(const Vector3f& camRadiance)
{
    // Radiance Factor shows the energy ratio between the path start point
    // and this location, so divison will give the radiance of that location
    totalRadiance += camRadiance / radFactor;
}

__device__ __forceinline__
void PathGuidingNode::AccumRadianceDownChain(const Vector3f& camRadiance,
                                             PathGuidingNode* gLocalChain)
{
    // Add to yourself
    AccumRadiance(camRadiance);
    for(IndexType i = prevNext[0]; i != InvalidIndex;)
    {
        gLocalChain[i].AccumRadiance(camRadiance);
        i = gLocalChain[i].prevNext[0];
    }
}

__device__ __forceinline__
void PathGuidingNode::AccumRadianceUpChain(const Vector3f& camRadiance,
                                           PathGuidingNode* gLocalChain)
{
    // Add to yourself
    AccumRadiance(camRadiance);
    for(IndexType i = prevNext[1]; i != InvalidIndex;)
    {
        gLocalChain[i].AccumRadiance(camRadiance);
        i = gLocalChain[i].prevNext[1];
    }
}