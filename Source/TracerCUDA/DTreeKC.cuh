﻿#pragma once

#include "RayLib/Vector.h"
#include "RayLib/CoordinateConversion.h"
#include "RayLib/Constants.h"
#include "RayLib/ColorConversion.h"

#include "RNGenerator.h"
#include "PathNode.cuh"
#include "DataStructsCommon.cuh"

struct DTreeNode
{
    enum NodeOrder
    {
        BOTTOM_LEFT = 0,
        BOTTOM_RIGHT = 1,
        TOP_LEFT = 2,
        TOP_RIGHT = 3
    };

    uint32_t                parentIndex;
    Vector4ui               childIndices;
    Vector4f                irradianceEstimates;
    Vector4ui               sampleCounts;

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
    DTreeNode*                  gRoot;
    uint32_t                    nodeCount;

    uint32_t                    totalSamples;
    float                       irradiance;

    __device__ Vector3f         Sample(float& pdf, RNGeneratorGPUI& rng) const;
    __device__ float            Pdf(const Vector3f& worldDir) const;
    __device__ void             AddRadianceToLeaf(const Vector3f& worldDir,
                                                  float radiance,
                                                  bool incrementSampleCount = false);
    __device__ void             AddSampleToLeaf(const Vector3f& worldDir);
};

__device__ inline
bool DTreeNode::IsRoot() const
{
    return parentIndex == UINT32_MAX;
}

__device__ inline
bool DTreeNode::IsLeaf(uint8_t childId) const
{
    return childIndices[childId] == UINT32_MAX;
}

__device__ inline
float DTreeNode::IrradianceEst(NodeOrder o) const
{
    // Do not return zero
    float irrad = irradianceEstimates[static_cast<int>(o)];
    return irrad;
}

__device__ inline
float DTreeNode::LocalPDF(uint8_t childIndex) const
{
    // Conditional PDF of x and y
    float pdfX, pdfY;
    pdfX = ((childIndex >> 0) & (0b01))
        ? (IrradianceEst(TOP_RIGHT) + IrradianceEst(BOTTOM_RIGHT))
        : (IrradianceEst(TOP_LEFT) + IrradianceEst(BOTTOM_LEFT));

    // Impossible to sample location
    if(pdfX == 0.0f) return 0.0f;

    pdfY = 1.0f / pdfX;
    pdfX /= irradianceEstimates.Sum();
    pdfY *= irradianceEstimates[childIndex];

    if(isnan(pdfX) || isinf(pdfX) ||
       isnan(pdfY) || isinf(pdfY))
        printf("pdfX %f, pdfY %f, childIndex %u\n",
               pdfX, pdfX, static_cast<uint32_t>(childIndex));

    return 4.0f * pdfX * pdfY;

    //return 4.0f * irradianceEstimates[childIndex] / irradianceEstimates.Sum();
}

__device__ inline
uint8_t DTreeNode::DetermineChild(const Vector2f& localCoords) const
{
    //assert(localCoords <= Vector3(1.0f) && localCoords >= Vector3(0.0f));
    if(localCoords > Vector3(1.0f) || localCoords < Vector3(0.0f))
    {
        printf("OUT OF RANGE LOCALS %f %f\n",
               localCoords[0], localCoords[1]);
    }

    uint8_t result = 0b00;
    if(localCoords[0] >= 0.5f) result |= (1 << 0) & (0b01);
    if(localCoords[1] >= 0.5f) result |= (1 << 1) & (0b10);
    return result;
}

__device__ inline
Vector2f DTreeNode::NormalizeCoordsForChild(uint8_t childIndex, const Vector2f& parentLocalCoords) const
{
    uint8_t isRight = ((childIndex >> 0) & 0b01) == 0b01;
    uint8_t isUp = ((childIndex >> 1) & 0b01) == 0b01;

    Vector2f localCoords = parentLocalCoords;
    if(isRight) localCoords[0] -= 0.5f;
    if(isUp)    localCoords[1] -= 0.5f;

    localCoords *= 2.0f;
    return localCoords;
}

__device__ inline
Vector3f DTreeGPU::Sample(float& pdf, RNGeneratorGPUI& rng) const
{
    Vector2f xi = Vector2f(rng.Uniform(), rng.Uniform());
    // First we need to find the sphr coords from the tree
    Vector2f discreteCoords = Zero2f;
    // Use double here for higher numeric precision for deep trees
    double descentFactor = 1.0;
    pdf = 1.0f;


    Vector2f initXi = xi;

    if(xi[0] < 0.0f || xi[0] >= 1.0f)
        printf("xi[0] fail from start xi: %f\n", xi[0]);
    if(xi[1] < 0.0f || xi[1] >= 1.0f)
        printf("xi[1] fail from start xi: %f\n", xi[1]);

    int i = 0;
    DTreeNode* node = gRoot;
    if(irradiance == 0.0f)
    {
        discreteCoords = xi;
    }
    else while(true)
    {
        Vector2f xiOld = xi;

        // Generate Local CDF
        float totalIrrad = (node->IrradianceEst(DTreeNode::BOTTOM_LEFT) +
                            node->IrradianceEst(DTreeNode::BOTTOM_RIGHT) +
                            node->IrradianceEst(DTreeNode::TOP_LEFT) +
                            node->IrradianceEst(DTreeNode::TOP_RIGHT));
        float totalIrradInverse = 1.0f / totalIrrad;
        float totalLeft = (node->IrradianceEst(DTreeNode::BOTTOM_LEFT) +
                            node->IrradianceEst(DTreeNode::TOP_LEFT));
        float totalRight = (node->IrradianceEst(DTreeNode::BOTTOM_RIGHT) +
                            node->IrradianceEst(DTreeNode::TOP_RIGHT));
        // CDF of X Axis
        float cdfMidX = totalLeft * totalIrradInverse;
        // CDF of Y Axis is depends on the selection of X
        float cdfMidY;

        float xi0Old = xi[0];

        uint8_t nextIndex = 0b00;
        // Locate X pos
        if(xi[0] < cdfMidX)
        {
            // Re-normalize sample for next iteration
            xi[0] = xi[0] / cdfMidX;

            if(isnan(xi[0]))
            {
                printf("L xi[0] is nan! at level %d, cdfMid %f, xi[0]:%f\n",
                       i, cdfMidX, xi0Old);
            }
        }
        else
        {

            // Re-normalize sample for next iteration
            //if((1.0f - cdfMidX) != 0.0f)
            //{
            xi[0] = (xi[0] - cdfMidX) / (1.0f - cdfMidX);
            //}
            //else xi[0] =

            if(isnan(xi[0]))
            {
                printf("R xi[0] is nan! at level %d, cdfMid %f, xi[0]:%f\n",
                       i, cdfMidX, xi0Old);
            }

            // Set the X bit on the iteration
            nextIndex |= (1 << 0) & (0b01);
        }

        // Determine Y CDF
        cdfMidY = ((nextIndex >> 0) & (0b01))
                   ? node->IrradianceEst(DTreeNode::BOTTOM_RIGHT) / totalRight
                   : node->IrradianceEst(DTreeNode::BOTTOM_LEFT) / totalLeft;

        // Locate Y Pos
        float xi1Old = xi[1];
        if(xi[1] < cdfMidY)
        {
            // Re-normalize sample for next iteration
            if(xi[1] == 0.0f && cdfMidY == 0.0f)
                printf("xi[1] Nan xi(%f, %f), midY %f, samples (%u, %u, %u, %u)\n",
                       xi[0], xi[1], cdfMidY,
                       node->sampleCounts[0], node->sampleCounts[1],
                       node->sampleCounts[2], node->sampleCounts[3]);

            xi[1] = xi[1] / cdfMidY;


            if(isnan(xi[1]))
            {
                printf("L xi[1] is nan! at level %d, cdfMid %f, xi[1]:%f\n", i,
                       cdfMidY, xi1Old);
            }
        }
        else
        {
            //if(xi[1] == 1.0f && cdfMidY == 1.0f)
            //    printf("xi[1] Nan xi(%f, %f), midY %f, samples (%u, %u, %u, %u)\n",
            //           xi[0], xi[1], cdfMidY,
            //           node->sampleCounts[0], node->sampleCounts[1],
            //           node->sampleCounts[2], node->sampleCounts[3]);

            // Re-normalize sample for next iteration
            xi[1] = (xi[1] - cdfMidY) / (1.0f - cdfMidY);
            // Set the Y bit on the iteration
            nextIndex |= (1 << 1) & (0b10);

            if(isnan(xi[1]))
            {
                printf("R xi[1] is nan! at level %d, cdfMid %f, xi[1]:%f\n", i,
                       cdfMidY, xi1Old);
            }
        }

        // Due to numerical precision, xi sometimes becomes one
        // just eliminate that case
        xi[0] = (xi[0] == 1.0f) ? nextafter(xi[0], 0.0f) : xi[0];
        xi[1] = (xi[1] == 1.0f) ? nextafter(xi[1], 0.0f) : xi[1];

        // Calculate current pdf and incorporate to the
        // main pdf conditionally
        if(xi[0] < 0.0f || xi[0] >= 1.0f)
            printf("%d: XI[0] OUT OF RANGE (%.10f => %.10f), cdf(%f, %f), NI: %u\n",
                   i, xiOld[0], xi[0], cdfMidX, cdfMidY,
                   static_cast<uint32_t>(nextIndex));
        if(xi[1] < 0.0f || xi[1] >= 1.0f)
            printf("%d: XI[1] OUT OF RANGE (%.10f => %.10f), cdf(%f, %f),  NI: %u\n",
                   i, xiOld[1], xi[1], cdfMidX, cdfMidY,
                   static_cast<uint32_t>(nextIndex));

        float localPDF = node->LocalPDF(nextIndex);
        pdf *= localPDF;

        Vector2f gridOffset(((nextIndex >> 0) & 0b01) ? 0.5f : 0.0f,
                            ((nextIndex >> 1) & 0b01) ? 0.5f : 0.0f);
        discreteCoords += gridOffset * descentFactor;
        descentFactor *= 0.5f;

        if(isnan(pdf) || discreteCoords.HasNaN())
            printf("pdf(%f) and/or DC(%f, %f) become "
                   "NAN at level %d, xi(%f, %f) mid(%f, %f) \n",
                   pdf, discreteCoords[0], discreteCoords[1], i,
                   xi[0], xi[1], cdfMidX, cdfMidY);

        //if(threadIdx.x == 0)
        //    printf("[%d] DC(%f, %f), pdf(%f), XI(%f, %f)\n",
        //            i, discreteCoords[0], discreteCoords[1],
        //            pdf, xi[0], xi[1]);

        if(node->IsLeaf(nextIndex))
        {
            // On leaf directly use sample as offset
            discreteCoords += xi * descentFactor;
            break;
        }
        node = gRoot + node->childIndices[nextIndex];
        i++;
    }

    if(isnan(pdf) || discreteCoords.HasNaN())
        printf("%d NAN? PDF(%f) DC(%f, %f) xi(%f, %f) initXi(%f, %f)\n",
               i, pdf,
               discreteCoords[0], discreteCoords[1],
               xi[0], xi[1], initXi[0], initXi[1]);

    Vector3f dir = GPUDataStructCommon::DiscreteCoordsToDir(pdf, discreteCoords);

    if(isnan(pdf) || dir.HasNaN())
        printf("%d NAN? PDF(%f) Dir(%f, %f, %f) xi(%f, %f)\n",
               i, pdf,
               dir[0], dir[1], dir[2],
               xi[0], xi[1]);

    return dir;
    //float discretePdf = pdf;
    //Vector3f result = TreeCoordsToWorldDir(pdf, discreteCoords);
    //pdf = discretePdf * 0.25f * MathConstants::InvPi;
    //if(threadIdx.x == 0)
    //    printf("Final dPDF %f, sPDF %f, DC (%f, %f), W (%f, %f, %f)\n",
    //           discretePdf, pdf,
    //           discreteCoords[0], discreteCoords[1],
    //           result[0], result[1], result[2]);
    //return result;
}

__device__ inline
float DTreeGPU::Pdf(const Vector3f& worldDir) const
{
    float pdf = 1.0f;
    Vector2f discreteCoords = GPUDataStructCommon::DirToDiscreteCoords(pdf, worldDir);

    DTreeNode* node = gRoot;
    Vector2f localCoords = discreteCoords;
    while(true)
    {
        uint8_t childIndex = node->DetermineChild(localCoords);
        pdf *= node->LocalPDF(childIndex);
        // Stop if leaf
        if(node->IsLeaf(childIndex)) break;
        // Continue Traversal
        localCoords = node->NormalizeCoordsForChild(childIndex, localCoords);
        node = gRoot + node->childIndices[childIndex];
    }
    return pdf;
}

__device__ inline
void DTreeGPU::AddRadianceToLeaf(const Vector3f& worldDir, float radiance,
                                 bool incrementSampleCount)
{
    Vector2f discreteCoords = GPUDataStructCommon::DirToDiscreteCoords(worldDir);
    //assert(discreteCoords >= Vector2f(0.0f) && discreteCoords <= Vector2f(1.0f));

    if(discreteCoords < Vector2f(0.0f) ||
       discreteCoords >= Vector2f(1.0f))
    {
        printf("Failed WorlDir = DC ((%f %f %f) = (%f %f))\n",
               worldDir[0], worldDir[1], worldDir[2],
               discreteCoords[0], discreteCoords[1]);
    }

    // Descent and find the leaf
    DTreeNode* node = gRoot;
    Vector2f localCoords = discreteCoords;

    while(true)
    {
        uint8_t childIndex = node->DetermineChild(localCoords);

        // If leaf atomically add the irrad value
        if(node->IsLeaf(childIndex))
        {
            if(isnan(radiance))
                printf("NaN Radiance add on node %lld\n",
                       static_cast<ptrdiff_t>(node - gRoot));

            //if(radiance != 0.0f)
            //    printf("%u  ", static_cast<uint32_t>(childIndex));

            atomicAdd(&node->irradianceEstimates[childIndex], radiance);
            if(incrementSampleCount)
                atomicAdd(&node->sampleCounts[childIndex], 1);
            break;
        }

        // Continue Traversal
        localCoords = node->NormalizeCoordsForChild(childIndex, localCoords);
        node = gRoot + node->childIndices[childIndex];
    }

    // Finally add a sample if required
    if(incrementSampleCount) atomicAdd(&totalSamples, 1);
}

__device__ inline
void DTreeGPU::AddSampleToLeaf(const Vector3f& worldDir)
{
    Vector2f discreteCoords = GPUDataStructCommon::DirToDiscreteCoords(worldDir);
    assert(discreteCoords <= Vector2f(1.0f) && discreteCoords >= Vector2f(0.0f));
    // Descent and find the leaf
    DTreeNode* node = gRoot;
    Vector2f localCoords = discreteCoords;
    while(true)
    {
        uint8_t childIndex = node->DetermineChild(localCoords);
        // If leaf atomically add the irrad value
        if(node->IsLeaf(childIndex))
        {
            atomicAdd(&node->sampleCounts[childIndex], 1);
            break;
        }
        // Continue Traversal
        localCoords = node->NormalizeCoordsForChild(childIndex, localCoords);
        node = gRoot + node->childIndices[childIndex];
    }
    // Also increment the total sample count of the tree
    atomicAdd(&totalSamples, 1);
}

__device__ inline
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
            // Do at-most minimal here only set the child id on the parent
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

__device__ inline
DTreeNode* PunchThroughNode(uint32_t& gNodeAllocLocation, DTreeGPU& gDTree,
                            const Vector2f& discreteCoords, uint32_t depth)
{
    // Go to the depth and allocate through the way
    uint32_t currentDepth = 0;
    Vector2f localCoords = discreteCoords;
    DTreeNode* node = gDTree.gRoot;
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
            uint32_t parentIndex = static_cast<uint32_t>(node - gDTree.gRoot);
            DTreeNode* childNode = gDTree.gRoot + childIndex;
            childNode->parentIndex = static_cast<uint32_t>(parentIndex);

            //printf("Punched Node %u \n", childIndex);
        }

        // Continue Traversal
        localCoords = node->NormalizeCoordsForChild(childId, localCoords);
        node = gDTree.gRoot + childIndex;
        currentDepth++;
    }
    //while(currentDepth <= depth);
    return node;
}

__device__ inline
void CalculateParentIrradiance(// I-O
                               DTreeGPU& gDTree,
                               // Input
                               uint32_t nodeIndex)
{
    DTreeNode* currentNode = gDTree.gRoot + nodeIndex;

    Vector4f& irrad = currentNode->irradianceEstimates;
    Vector4ui& sampleCounts = currentNode->sampleCounts;
    // Omit zeros on the tree & average the value
    auto AverageIrrad = [&](uint32_t i)
    {
        if(!currentNode->IsLeaf(i)) return;
        irrad[i] = (sampleCounts[i] == 0)
                    ? 0.0f
                    //: irrad[i] / static_cast<float>(sampleCounts[i]);
                    : irrad[i];
        irrad[i] = max(irrad[i], MathConstants::VeryLargeEpsilon);
    };
    AverageIrrad(0);
    AverageIrrad(1);
    AverageIrrad(2);
    AverageIrrad(3);

    // Total leaf irrad
    float sum = 0.0f;
    sum += currentNode->IsLeaf(0) ? irrad[0] : 0.0f;
    sum += currentNode->IsLeaf(1) ? irrad[1] : 0.0f;
    sum += currentNode->IsLeaf(2) ? irrad[2] : 0.0f;
    sum += currentNode->IsLeaf(3) ? irrad[3] : 0.0f;

    // Back-propagate the sum towards the root
    DTreeNode* n = currentNode;
    while(!(n->IsRoot()))
    {
        DTreeNode* parentNode = gDTree.gRoot + n->parentIndex;
        uint32_t nodeIndex = static_cast<uint32_t>(n - gDTree.gRoot);

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

    // Finally add to the total as well
    atomicAdd(&gDTree.irradiance, sum);
}

//__device__ inline
//void CalculateMaxDepth(// Output
//                       uint32_t& gMaxDepth,
//                       // Input
//                       const DTreeGPU& gDTree,
//                       uint32_t nodeIndex)
//{
//    const DTreeNode& gMyNode = gDTree.gRoot[nodeIndex];
//    // Only launch the loop if the all children are leafs
//    // since those will generate the max depth
//    bool doMax = true;
//    doMax &= gMyNode.IsLeaf(0);
//    doMax &= gMyNode.IsLeaf(1);
//    doMax &= gMyNode.IsLeaf(2);
//    doMax &= gMyNode.IsLeaf(3);
//
//    if(doMax)
//    {
//        // Leaf -> Root Traverse (Bottom up)
//        uint32_t depth = 1;
//        for(const DTreeNode* curNode = &gMyNode;
//            !curNode->IsRoot();
//            curNode = gDTree.gRoot + curNode->parentIndex)
//        {
//            depth++;
//        }
//        // Atomic MAX DEPTH
//        // Normally i should implement an transform_reduce function for gpu
//        // but this works and simpler to implement
//        atomicMax(&gMaxDepth, depth);
//    }
//}
//
//__device__ inline
//void NormalizeIrradiances(// I-O
//                          DTreeGPU& gDTree,
//                          uint32_t maxDepth,
//                          uint32_t nodeIndex)
//{
//    const DTreeNode& gMyNode = gDTree.gRoot[nodeIndex];
//    // Leaf -> Root Traverse (Bottom up)
//    uint32_t depth = 1;
//    for(const DTreeNode* curNode = &gMyNode;
//        !curNode->IsRoot();
//        curNode = gDTree.gRoot + curNode->parentIndex)
//    {
//        depth++;
//    }
//    // Normalization factor
//    float factor = static_cast<float>(1 << (2 * (maxDepth - depth)));
//    gDTree.gRoot[nodeIndex].irradianceEstimates[0] *= factor;
//    gDTree.gRoot[nodeIndex].irradianceEstimates[1] *= factor;
//    gDTree.gRoot[nodeIndex].irradianceEstimates[2] *= factor;
//    gDTree.gRoot[nodeIndex].irradianceEstimates[3] *= factor;
//}

__device__ inline
void MarkChildRequest(// Output
                      uint32_t* gRequestedChilds,
                      // Input
                      const DTreeGPU& gDTree,
                      float fluxRatio,
                      uint32_t depthLimit,
                      uint32_t nodeIndex)
{
    uint32_t depth = 0;
    DTreeNode* n = gDTree.gRoot + nodeIndex;
    while(!(n->IsRoot()))
    {
        DTreeNode* parentNode = gDTree.gRoot + n->parentIndex;
        // Traverse upwards
        n = parentNode;
        depth++;
    }

    //printf("[%p] n:%u, myDepth %u\n",
    //       gDTree.gRoot, nodeIndex, depth);

    // We can't create anymore children depth limit reached
    if(depth >= depthLimit)
    {
        gRequestedChilds[nodeIndex] = 0;
        return;
    }

    // Kernel Grid - Stride Loop
    DTreeNode* node = gDTree.gRoot + nodeIndex;
    uint32_t requestedChildCount = 0;
    if(gDTree.irradiance > 0.0f)
    {
        // Check if we need child
        UNROLL_LOOP
        for(uint32_t i = 0; i < 4; i++)
        {
            float localIrrad = node->irradianceEstimates[i];
            float percentFlux = localIrrad / gDTree.irradiance;

            //printf("%u : Mark child! Local %f Total %f Percent %f\n",
            //       threadId,
            //       localIrrad, gDTree.irradiance, percentFlux);

            if(percentFlux > fluxRatio)
                requestedChildCount++;
        }
    }
    gRequestedChilds[nodeIndex] = requestedChildCount;
}

__device__ inline
void ReconstructEmptyTree(// Output
                          DTreeGPU& gDTree,
                          uint32_t& gDTreeAllocator,
                          // Input
                          const DTreeGPU& gSiblingTree,
                          uint32_t allocatedNodeCount,
                          float fluxRatio,
                          uint32_t depthLimit,
                          uint32_t nodeIndex)
{
    DTreeNode* siblingNode = gSiblingTree.gRoot + nodeIndex;
    float localIrrad = siblingNode->irradianceEstimates.Sum();
    float percentFlux = localIrrad / gSiblingTree.irradiance;

    // Skip if there is no need for this node on the next tree
    if(gSiblingTree.irradiance == 0.0f ||
        percentFlux <= fluxRatio)
        return;

    //printf("%u : Split wanted Local %f Total %f Percent %f\n",
    //       threadIdx.x + blockDim.x * blockIdx.x,
    //       localIrrad, gSiblingTree.irradiance, percentFlux);

    // Generate discrete point and a depth for tree traversal
    // Use parent pointers to determine your node coords and depth
    int depth = 0;
    Vector2f discretePoint = Vector2f(0.5f);

    DTreeNode* n = siblingNode;
    while(!(n->IsRoot()))
    {
        DTreeNode* parentNode = gSiblingTree.gRoot + n->parentIndex;
        uint32_t nodeIndex = static_cast<uint32_t>(n - gSiblingTree.gRoot);

        uint32_t childId = UINT32_MAX;
        childId = (parentNode->childIndices[0] == nodeIndex) ? 0 : childId;
        childId = (parentNode->childIndices[1] == nodeIndex) ? 1 : childId;
        childId = (parentNode->childIndices[2] == nodeIndex) ? 2 : childId;
        childId = (parentNode->childIndices[3] == nodeIndex) ? 3 : childId;

        Vector2f childCoordOffset(((childId >> 0) & 0b01) ? 0.5f : 0.0f,
                                  ((childId >> 1) & 0b01) ? 0.5f : 0.0f);
        discretePoint = childCoordOffset + 0.5f * discretePoint;
        depth++;

        // Traverse upwards
        n = parentNode;
    }

    // Do not create this node if it is over depth limit
    if(depth > depthLimit)
    {
        printf("depthLimit-normal-reached\n");
        return;
    }

    //printf("My Point %f %f \n", discretePoint[0], discretePoint[1]);

    // Punch-through this node to the new tree
    // Meaning, traverse and allocate (if not already allocated)
    // until we created the equivalent node
    DTreeNode* punchedNode = PunchThroughNode(gDTreeAllocator, gDTree,
                                                discretePoint, depth);
    uint32_t punchedNodeId = static_cast<uint32_t>(punchedNode - gDTree.gRoot);

    // Do not create children if children over depth limit
    if((depth + 1) > depthLimit)
    {
        //printf("depthLimit-forchild-reached\n");
        return;
    }
    // We allocated up to this point
    // Check children if they need allocation
    uint8_t childCount = 0;
    Vector4uc childOffsets = Vector4uc(UINT8_MAX);
    UNROLL_LOOP
    for(uint32_t i = 0; i < 4; i++)
    {
        float localIrrad = siblingNode->irradianceEstimates[i];
        float percentFlux = localIrrad / gSiblingTree.irradiance;

        // Iff create if node was not available on the sibling tree
        if(percentFlux > fluxRatio && siblingNode->IsLeaf(i))
        {
            childOffsets[i] = childCount;
            childCount++;
        }
    }

    // Allocate children
    uint32_t childGlobalOffset = atomicAdd(&gDTreeAllocator, childCount);

    if(allocatedNodeCount < childCount + childGlobalOffset)
    {
        printf("Overallocating children (depth %u), skipping!\n", depth);
        assert(false);
        return;
    }

    //printf("Child Count %u, Offsets %u %u %u %u\n",
    //       static_cast<uint32_t>(childCount),
    //       static_cast<uint32_t>(childOffsets[0]),
    //       static_cast<uint32_t>(childOffsets[1]),
    //       static_cast<uint32_t>(childOffsets[2]),
    //       static_cast<uint32_t>(childOffsets[3]));

    UNROLL_LOOP
    for(uint32_t i = 0; i < 4; i++)
    {
        //float localIrrad = siblingNode->irradianceEstimates[i];
        //float percentFlux = localIrrad / gSiblingTree.irradiance;
        //if(percentFlux > fluxRatio && siblingNode->IsLeaf(i))
        if(childOffsets[i] != UINT8_MAX)
        {
            // Allocate a new node
            uint32_t childNodeIndex = childGlobalOffset + childOffsets[i];
            DTreeNode* childNode = gDTree.gRoot + childNodeIndex;

            // Set the punched node's child id
            punchedNode->childIndices[i] = childNodeIndex;
            childNode->parentIndex = static_cast<uint32_t>(punchedNodeId);

            //printf("Creating Child %u, local %u \n", childNodeIndex,
            //       static_cast<uint32_t>(i));
        }
    }
    // All Done!
}