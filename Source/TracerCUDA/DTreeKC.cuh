#pragma once

#include "RayLib/Vector.h"
#include "RayLib/CoordinateConversion.h"
#include "RayLib/Constants.h"

#include "Random.cuh"


struct DTreeNode
{
    enum NodeOrder
    {
        BOTTOM_LEFT = 0,
        BOTTOM_RIGHT = 1,
        TOP_LEFT = 2,
        TOP_RIGHT = 3
    };

    uint16_t                parentIndex;
    Vector4ui               childIndices;
    Vector4f                irradianceEstimates;

    __device__ bool         IsRoot() const;
    __device__ bool         IsLeaf(uint8_t childId) const;
    __device__ float        IrradianceEst(NodeOrder) const;
    __device__ uint8_t      DetermineChild(const Vector2f& localCoords) const;
    __device__ Vector2f     NormalizeCoordsForChild(uint8_t childIndex,
                                                    const Vector2f& parentLocalCoords) const;
    __device__ float        LocalPDF(uint8_t childIndex) const;
};

struct DTreeGPU
{
    DTreeNode*              gRoot;
    uint32_t                nodeCount;

    uint32_t                totalSamples;
    float                   irradiance;

    __device__ Vector3f     Sample(float& pdf, RandomGPU& rng) const;
    __device__ float        Pdf(const Vector3f& worldDir) const;

    __device__ void         AddRadianceToLeaf(const Vector3f& worldDir,
                                              float radiance);
};

__device__ __forceinline__
bool DTreeNode::IsRoot() const
{
    return parentIndex == UINT16_MAX;
}

__device__ __forceinline__
bool DTreeNode::IsLeaf(uint8_t childId) const
{
    return childIndices[childId] == UINT32_MAX;
}

__device__ __forceinline__
float DTreeNode::IrradianceEst(NodeOrder o) const
{
    return irradianceEstimates[static_cast<int>(o)];
}

__device__ __forceinline__
float DTreeNode::LocalPDF(uint8_t childIndex) const
{
    // I do not understand where this 4 is coming from
    return 4 * irradianceEstimates[childIndex] / irradianceEstimates.Sum();
}

__device__ __forceinline__
uint8_t DTreeNode::DetermineChild(const Vector2f& localCoords) const
{
    uint8_t result = 0b00;
    if(localCoords[0] > 0.5f) result |= (1 << 0) & (0b01);
    if(localCoords[1] > 0.5f) result |= (1 << 1) & (0b10);
    return result;
}

__device__ __forceinline__
Vector2f DTreeNode::NormalizeCoordsForChild(uint8_t childIndex, const Vector2f& parentLocalCoords) const
{
    uint8_t isLeft = (childIndex & 0b01) == 0b0;
    uint8_t isDown = ((childIndex >> 1) & 0b01) == 0b0;

    Vector2f localCoords = parentLocalCoords;
    if(!isLeft) localCoords[0] -= 0.5f;
    if(!isDown) localCoords[1] -= 0.5f;

    localCoords *= 2.0f;
    return localCoords;
}

__device__ __forceinline__
Vector3f DTreeGPU::Sample(float& pdf, RandomGPU& rng) const
{
    Vector2f xi = Vector2f(GPUDistribution::Uniform<float>(rng),
                           GPUDistribution::Uniform<float>(rng));

    // First we need to find the sphr coords from the tree
    Vector2f discreteCoords = Zero2f;
    // Use double here for higher numeric precision
    double descentFactor = 1;

    pdf = 1.0f;
    DTreeNode* node = gRoot;
    do
    {
        // Generate Local CDF
        float totalIrrad = (node->IrradianceEst(DTreeNode::BOTTOM_LEFT) +
                            node->IrradianceEst(DTreeNode::BOTTOM_RIGHT) +
                            node->IrradianceEst(DTreeNode::TOP_LEFT) +
                            node->IrradianceEst(DTreeNode::BOTTOM_RIGHT));
        float totalIrradInverse = 1.0f / totalIrrad;
            
        // Generate a 2 data CDF for determine the sample
        // with these we will do the inverse sampling
        // only split points are required since the other CDF data will
        // implcitly be one
        float cdfMidX = (node->IrradianceEst(DTreeNode::BOTTOM_LEFT) +
                         node->IrradianceEst(DTreeNode::TOP_LEFT)) * totalIrradInverse;
        float cdfMidY = (node->IrradianceEst(DTreeNode::BOTTOM_LEFT) +
                         node->IrradianceEst(DTreeNode::BOTTOM_RIGHT)) * totalIrradInverse;
                
        uint8_t nextIndex = 0b00;
        // Locate X pos
        if(xi[0] < cdfMidX)
        {
            // Renormalize sample for next iteration
            xi[0] = xi[0] / cdfMidX;
        }
        else
        {
            // Renormalize sample for next iteration
            xi[0] = (xi[0] - cdfMidX) / (1.0f - cdfMidX);
            // Set the X bit on the iteration
            nextIndex |= (1 << 0) & (0b01);
        }
        // Locate Y Pos
        if(xi[1] < cdfMidY)
        {
            // Renormalize sample for next iteration
            xi[1] = xi[1] / cdfMidY;
        }
        else
        {
            // Renormalize sample for next iteration
            xi[1] = (xi[1] - cdfMidY) / (1.0f - cdfMidY);
            // Set the X bit on the iteration
            nextIndex |= (1 << 1) & (0b10);            
        }

        // Calculate current pdf and incorporate to the
        // main pdf conditionally
        pdf *= node->LocalPDF(nextIndex);


        Vector2f gridOffset(((nextIndex >> 0) & 0b01) ? 0.5f : 0.0f,
                            ((nextIndex >> 1) & 0b01) ? 0.5f : 0.0f);
        discreteCoords += gridOffset * descentFactor;
        descentFactor *= 0.5;

        if(node->IsLeaf(nextIndex))
        {
            // On leaf directly use sample as offset
            discreteCoords += xi * descentFactor;
            break;
        }
        node = gRoot + node->childIndices[nextIndex];
        
    }
    while(true);

    // Convert PDF to Solid Angle PDF
    pdf *= 0.25f * MathConstants::InvPi;

    // Convert the Local 2D cartesian coords to spherical coords
    Vector2f sphrCoords = Vector2f(2.0f * discreteCoords[0],
                                   (2.0f * discreteCoords[1] - 1.0f));
    sphrCoords *= MathConstants::Pi;
    // Then convert spherical coords to 3D cartesian world space coords
    return Utility::SphericalToCartesianUnit(sphrCoords);
}

__device__ __forceinline__
float DTreeGPU::Pdf(const Vector3f& worldDir) const
{
    Vector2f sphrCoords = Utility::CartesianToSphericalUnit(worldDir);
    Vector2f discreteCoords = sphrCoords * MathConstants::InvPi;
    discreteCoords = discreteCoords + Vector2f(0.0f, 1.0f);
    discreteCoords *= 0.5f;

    float pdf = 1.0f;
    DTreeNode* node = gRoot;
    Vector2f localCoords = discreteCoords;
    do
    {
        uint8_t childIndex = node->DetermineChild(localCoords);

        float totalIrrad = (node->IrradianceEst(DTreeNode::BOTTOM_LEFT) +
                            node->IrradianceEst(DTreeNode::BOTTOM_RIGHT) +
                            node->IrradianceEst(DTreeNode::TOP_LEFT) +
                            node->IrradianceEst(DTreeNode::BOTTOM_RIGHT));

        

        pdf *= node->LocalPDF(childIndex);

        // Stop if leaf
        if(node->IsLeaf(childIndex)) break;
        // Continue Traversal
        localCoords = node->NormalizeCoordsForChild(childIndex, localCoords);
        node = gRoot + node->childIndices[childIndex];
    }
    while(true);

    // Convert PDF to Solid Angle PDF
    pdf *= 0.25f * MathConstants::InvPi;
    return pdf;
}

__device__ __forceinline__
void DTreeGPU::AddRadianceToLeaf(const Vector3f& worldDir, float radiance)
{
    Vector2f sphrCoords = Utility::CartesianToSphericalUnit(worldDir);
    Vector2f discreteCoords = sphrCoords * MathConstants::InvPi;
    discreteCoords = discreteCoords + Vector2f(0.0f, 1.0f);
    discreteCoords *= 0.5f;

    // Descent and find the leaf
    DTreeNode* node = gRoot;
    Vector2f localCoords = discreteCoords;
    do
    {
        uint8_t childIndex = node->DetermineChild(localCoords);

        // If leaf atomically add the irrad value
        if(node->IsLeaf(childIndex))
        {
            atomicAdd(&node->irradianceEstimates[childIndex], radiance);
            break;
        }

        // Continue Traversal
        localCoords = node->NormalizeCoordsForChild(childIndex, localCoords);
        node = gRoot + node->childIndices[childIndex];
    }
    while(true);
}

__device__ __forceinline__ 
uint32_t AtomicAllocateNode(bool& allocated, 
                            uint8_t childId, 
                            DTreeNode* gParentNode, 
                            uint32_t& gAllocator)
{    
    allocated = false;
    // 0xFFFFFFFF means empty (non-allocated) node
    static constexpr uint32_t EMPTY_NODE = UINT32_MAX;
    // 0xFFFFFFFE means allocation in progress
    static constexpr uint32_t ALLOCATING = UINT32_MAX - 1;
    // All other numbers are valid nodes 
    // (unless of course those are out of bounds)

    // Just take node if already allocated
    // Just do a pre-check in order to skip atomic stuff
    if(gParentNode->childIndices[childId] < ALLOCATING) 
        return gParentNode->childIndices[childId];

    // Try to lock the node and allocate for that node
    uint32_t old = ALLOCATING;
    while(old == ALLOCATING)
    {
        old = atomicCAS(&gParentNode->childIndices[childId], EMPTY_NODE, ALLOCATING);
        if(old == EMPTY_NODE)
        {
            // This thread is selected to actually allocate
            // Do atmost minimal here only set the child id on the parent
            // We are allocating in a top-down fashion set other memory stuff later
            uint32_t location = static_cast<uint32_t>(atomicAdd(&gAllocator, 1u));
            reinterpret_cast<volatile uint32_t&>(gParentNode->childIndices[childId]) = location;
            old = location;

            // Just flag here do the rest after
            allocated = true;
        }
        // This is important somehow compiler changes this and makes infinite loop on same warp threads
        __threadfence();	
    }
    return old;
}

__device__ __forceinline__
DTreeNode* PunchThroughNode(uint32_t& gNodeAllocLocation, DTreeGPU* gDTree,
                            const Vector2f& discreteCoords, uint32_t depth)
{
    // Go to the depth and allocate through the way
    uint32_t currentDepth = 0;
    Vector2f localCoords = discreteCoords;
    DTreeNode* node = gDTree->gRoot;
    while(currentDepth < depth)
    {
        uint8_t childId = node->DetermineChild(localCoords);

        // Atomically Allocate a node from the array
        bool allocated;
        uint32_t childIndex = AtomicAllocateNode(allocated, childId, node, gNodeAllocLocation);
        
        // This thread is allocated the node
        // Do the rest of the initialization work here
        if(allocated)
        {
            uint32_t parentIndex = static_cast<uint32_t>(node - gDTree->gRoot);
            DTreeNode* childNode = gDTree->gRoot + childIndex;
            childNode->parentIndex = static_cast<uint16_t>(parentIndex);
        }

        // Continue Traversal
        localCoords = node->NormalizeCoordsForChild(childId, localCoords);
        node = gDTree->gRoot + node->childIndices[childId];
        currentDepth++;
    }
    //while(currentDepth <= depth);
    return node;
}

__global__
static void KCCalculateParentIrradiance(// I-O
                                        DTreeGPU* gDTree,
                                        // Input
                                        uint32_t totalNodeCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalNodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        DTreeNode* currentNode = gDTree->gRoot + threadId;
        
        // Total leaf irrad        
        float sum = 0.0f;
        sum += currentNode->IsLeaf(0) ? currentNode->irradianceEstimates[0] : 0.0f;
        sum += currentNode->IsLeaf(1) ? currentNode->irradianceEstimates[1] : 0.0f;
        sum += currentNode->IsLeaf(2) ? currentNode->irradianceEstimates[2] : 0.0f;
        sum += currentNode->IsLeaf(3) ? currentNode->irradianceEstimates[3] : 0.0f;

        // Back-propogate the sum towards the root
        DTreeNode* n = currentNode;
        while(!(n->IsRoot()))
        {
            DTreeNode* parentNode = gDTree->gRoot + n->parentIndex;
            uint32_t nodeIndex = static_cast<uint32_t>(n - gDTree->gRoot);

            uint32_t childId = UINT32_MAX;
            childId = (parentNode->childIndices[0] == nodeIndex) ? 0 : childId;
            childId = (parentNode->childIndices[1] == nodeIndex) ? 1 : childId;
            childId = (parentNode->childIndices[2] == nodeIndex) ? 2 : childId;
            childId = (parentNode->childIndices[3] == nodeIndex) ? 3 : childId;

            // Atomically add since other threads will add to the estimate
            atomicAdd(&parentNode->irradianceEstimates[childId], sum);
            // Traverse upwards
            n = parentNode;
        }

        // Finally add to the total aswell
        atomicAdd(&gDTree->irradiance, sum);        
    }
}

__global__
static void KCMarkChildRequest(// Output
                               uint32_t* gRequestedChilds,
                               // Input               
                               const DTreeGPU* gDTree,
                               float fluxRatio,
                               uint32_t totalNodeCount)
{
    // Kernel Grid - Stride Loop
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalNodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        DTreeNode* node = gDTree->gRoot + threadId;
        uint32_t requestedChildCount = 0;
        // Check if we need
        UNROLL_LOOP
        for(uint32_t i = 0; i < 4; i++)
        {
            float localIrrad = node->irradianceEstimates[i];
            float percentFlux = localIrrad / gDTree->irradiance;

            if(percentFlux > fluxRatio)
                requestedChildCount++;             
        }       
        gRequestedChilds[threadId] = requestedChildCount;
    }
}

__global__
static void KCReconstructEmptyTree(// Output
                                   DTreeGPU* gDTree,
                                   // Input               
                                   const DTreeGPU* gSiblingTree,
                                   float fluxRatio,
                                   uint32_t depthLimit,
                                   uint32_t siblingNodeCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < siblingNodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        DTreeNode* siblingNode = gSiblingTree->gRoot + threadId;
        float localIrrad = siblingNode->irradianceEstimates.Sum();
        float percentFlux = localIrrad / gDTree->irradiance;

        // Skip if there is no need for this node on the next tree
        if(percentFlux <= fluxRatio) continue;

        // Generate discrete point and a depth for tree traversal
        // Use parent pointers to determine your node coords and depth
        int depth = 0; 
        Vector2f discretePoint = Vector2f(0.5f);

        DTreeNode* n = siblingNode;
        while(!(n->IsRoot()))
        {
            DTreeNode* parentNode = gSiblingTree->gRoot + n->parentIndex;
            uint32_t nodeIndex = static_cast<uint32_t>(n - gSiblingTree->gRoot);

            uint32_t childId = UINT32_MAX;
            childId = (parentNode->childIndices[0] == nodeIndex) ? 0 : childId;
            childId = (parentNode->childIndices[1] == nodeIndex) ? 1 : childId;
            childId = (parentNode->childIndices[2] == nodeIndex) ? 2 : childId;
            childId = (parentNode->childIndices[3] == nodeIndex) ? 3 : childId;

            Vector2f childCoordOffset(((childId >> 0) & 0b01) ? 0.5f : 0.0f,
                                      ((childId >> 1) & 0b01) ? 0.5f : 0.0f);
            discretePoint += childCoordOffset + 0.5f * discretePoint;
            depth++;

            // Traverse upwards
            n = parentNode;
        }

        // Punchthrough this node to the new tree
        // Meaning, traverse and allocate (if not already allocated)
        // until we created the equavilent node
        DTreeNode* punchedNode = PunchThroughNode(gDTree->nodeCount, gDTree,
                                                  discretePoint, depth);
        uint32_t punchedNodeId = static_cast<uint32_t>(punchedNode - gDTree->gRoot);
      

        printf("PunchId %u\n", punchedNodeId);

        // We allocated up to this point
        // Check childs if they need allocation
        for(uint32_t i = 0; i < 4; i++)
        {
            float localIrrad = siblingNode->irradianceEstimates[i];
            float percentFlux = localIrrad / gSiblingTree->irradiance;
            
            if(percentFlux > fluxRatio &&
               // Iff create if node was not available on the sibling tree
               siblingNode->IsLeaf(i))
            {
                // Allocate a new node
                uint32_t childNodeIndex = atomicAdd(&gDTree->nodeCount, 1);
                DTreeNode* childNode = gDTree->gRoot + childNodeIndex;
                childNode->parentIndex = static_cast<uint16_t>(punchedNodeId);
                punchedNode->childIndices[i] = childNodeIndex;

                printf("DOING STUFF\n");
            }
        }
        // All Done!
    }
}

__global__ 
static void KCInitDTreeNodes(DTreeGPU* gTree, uint32_t nodeCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < nodeCount;
        threadId += (blockDim.x * gridDim.x))
    {
        DTreeNode node;
        node.irradianceEstimates = Zero4;
        node.parentIndex = UINT16_MAX;
        node.childIndices = Vector4ui(UINT32_MAX);

        gTree->gRoot[threadId] = node;

        if(threadId == 0)
        {
            gTree->irradiance = 0.0f;
            gTree->totalSamples = 0;
            // Root node is always assumed to be allocated
            gTree->nodeCount = 1;
        }
    }
}