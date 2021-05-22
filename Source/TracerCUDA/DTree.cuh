#pragma once

/**

DTree Implementation

*/ 

#include "RayLib/Vector.h"
#include "RayLib/CoordinateConversion.h"
#include "RayLib/Constants.h"

#include "Random.cuh"
#include "DeviceMemory.h"

struct DTreeNode
{
    enum NodeOrder
    {
        BOTTOM_LEFT     = 0,
        BOTTOM_RIGHT    = 1,
        TOP_LEFT        = 2,
        TOP_RIGHT       = 3
    };

    uint32_t                childIndex;
    Vector4f                irradianceEstimates;

    __device__ bool         IsLeaf() const;
    __device__ float        IrradianceEst(NodeOrder) const;
    __device__ uint8_t      DetermineChild(const Vector2f& localCoords) const;
    __device__ Vector2f     NormalizeCoordsForChild(const Vector2f& parentLocalCoords) const;
};

struct DTreeGPU
{
    DTreeNode*  gDTreeNodes;

    uint32_t    totalSamples;
    float       irradiance;
    
    __device__ Vector3f Sample(float& pdf, RandomGPU& rng) const;
    __device__ float Pdf(const Vector3f& worldDir) const;
};

class DTree
{
    private:
        DeviceMemory    treeNodeMemory;
        DTreeNode*      dTreeNodes;

    protected:

    public:
        // Constructors & Destructor
                        DTree();
                        DTree(const DTree&) = delete;
                        DTree(DTree&&) = default;
        DTree&          operator=(const DTree&) = delete;
        DTree&          operator=(DTree&&) = default;
                        ~DTree() = default;

        // Members
        
};

__device__ __forceinline__
bool DTreeNode::IsLeaf() const
{
    return childIndex == UINT32_MAX;
}

__device__ float DTreeNode::IrradianceEst(NodeOrder o) const
{
    return irradianceEstimates[static_cast<int>(o)];
}

__device__ __forceinline__
uint8_t DTreeNode::DetermineChild(const Vector2f& localCoords) const
{
  
}

__device__ __forceinline__
Vector2f DTreeNode::NormalizeCoordsForChild(uint8_t childIndex, const Vector2f& parentLocalCoords) const
{

}

__device__ __forceinline__
Vector3f DTreeGPU::Sample(float& pdf, RandomGPU& rng) const
{
    Vector2f xi = Vector2f(GPUDistribution::Uniform<float>(rng),
                           GPUDistribution::Uniform<float>(rng));

    // First we need to find the sphr coords from the tree
    Vector2f discreteCoords = Zero2f;

    pdf = 1.0f;
    DTreeNode* node = gDTreeNodes;
    do
    {
        // Generate Local CDF
        float totalIrrad = (node->IrradianceEst(DTreeNode::BOTTOM_LEFT) +
                            node->IrradianceEst(DTreeNode::BOTTOM_LEFT) +
                            node->IrradianceEst(DTreeNode::TOP_LEFT) +
                            node->IrradianceEst(DTreeNode::BOTTOM_LEFT));
        float totalIrradInverse = 1.0f / totalIrrad;
            
        float cdfMidX = (node->IrradianceEst(DTreeNode::BOTTOM_LEFT) +
                         node->IrradianceEst(DTreeNode::TOP_LEFT)) * totalIrradInverse;
        float cdfMidY = (node->IrradianceEst(DTreeNode::BOTTOM_LEFT) +
                         node->IrradianceEst(DTreeNode::BOTTOM_RIGHT)) * totalIrradInverse;
        
        uint8_t nextIndex = 0b00;
        if(node->IsLeaf())
        {
            .......
            break;
        }
        else
        {
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
        }

        node = gDTreeNodes + node->childIndex + nextIndex;
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
    DTreeNode* node = gDTreeNodes;
    do
    {

    }
    while(true);

    // Convert PDF to Solid Angle PDF
    pdf *= 0.25f * MathConstants::InvPi;
}