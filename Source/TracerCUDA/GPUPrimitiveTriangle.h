#pragma once
/**

Default Triangle Implementation

Has three types of data
Position, Normal and UV.

These data are indirected by a single index (like DirectX and OpenGL)

All of them should be provided

*/

#include <map>
#include <tuple>
#include <vector>

#include "RayLib/PrimitiveDataTypes.h"
#include "RayLib/Vector.h"
#include "RayLib/Triangle.h"
#include "RayLib/CoordinateConversion.h"
#include "RayLib/MortonCode.h"

#include "RNGenerator.h"
#include "GPUPrimitiveP.cuh"
#include "BinarySearch.cuh"

#include "GPUTransformI.h"
#include "GPUSurface.h"
#include "DefaultLeaf.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"
#include "TextureFunctions.h"


class SurfaceDataLoaderI;
using SurfaceDataLoaders = std::vector<std::unique_ptr<SurfaceDataLoaderI>>;

// Triangle Memory Layout
struct TriData
{
    // Kinda Perf Hog but most memory efficient
    // Binary search cull face for each prim
    const GPUBitmap**   alphaMaps;
    const bool*         cullFace;
    const uint64_t*     primOffsets;
    uint32_t            primBatchCount;

    // Per vertex attributes
    const Vector3f*     positions;
    const QuatF*        tbnRotations;
    const Vector2*      uvs;
    // Single indexed vertices
    const uint64_t*     indexList;
};

// Triangle Hit is barycentric coordinates
// c is (1-a-b) thus it is not stored.
using TriangleHit = Vector2f;

struct TriFunctions
{
    __device__ inline
    static Vector3f SamplePosition(// Output
                                   Vector3f& normal,
                                   float& pdf,
                                   // Input
                                   const GPUTransformI& transform,
                                   //
                                   PrimitiveId primitiveId,
                                   const TriData& primData,
                                   // I-O
                                   RNGeneratorGPUI& rng)
    {
        Vector2f xi = rng.Uniform2D();
        float r1 = sqrt(xi[0]);
        float r2 = xi[1];
        // Generate Random Barycentrics
        // Osada 2002
        // http://graphics.stanford.edu/courses/cs468-08-fall/pdf/osada.pdf
        float a = 1 - r1;
        float b = (1 - r2) * r1;
        float c = r1 * r2;

        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        // Calculate PDF
        // Approximate the area with the determinant
        float area = TriFunctions::Area(transform, primitiveId, primData);
        pdf = 1.0f / area;

        // Calculate Normal
        // CCW
        QuatF q0 = primData.tbnRotations[index0].Normalize();
        QuatF q1 = primData.tbnRotations[index1].Normalize();
        QuatF q2 = primData.tbnRotations[index2].Normalize();
        QuatF tbn = Quat::BarySLerp(q0, q1, q2, a, b);
        Vector3 Z_AXIS = ZAxis;

        normal = tbn.Conjugate().ApplyRotation(Z_AXIS);
        Vector3 position = (position0 * a +
                            position1 * b +
                            position2 * c);

        normal = transform.LocalToWorld(normal, true).Normalize();
        position = transform.LocalToWorld(position);

        return position;
    }

    __device__ inline
    static void PositionPdfFromReference(// Outputs
                                         Vector3f& normal,
                                         float& pdf,
                                         float& distance,
                                         // Inputs
                                         const RayF& ray,
                                         const GPUTransformI& transform,
                                         //
                                         const PrimitiveId primitiveId,
                                         const TriData& primData)
    {
        // Find the primitive
        float index;
        GPUFunctions::BinarySearchInBetween(index, primitiveId,
                                            primData.primOffsets,
                                            primData.primBatchCount);
        uint32_t indexInt = static_cast<uint32_t>(index);
        bool cullBackface = primData.cullFace[indexInt];

        // Find the primitive
        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        RayF r = ray;
        r = transform.WorldToLocal(r);

        Vector3 baryCoords;
        bool intersects = r.IntersectsTriangle(baryCoords, distance,
                                               position0,
                                               position1,
                                               position2,
                                               cullBackface);

        // Check if an alpha map exists and accept/reject intersection
        const GPUBitmap* alphaMap = primData.alphaMaps[indexInt];
        if(alphaMap && intersects)
        {
            Vector2f uv0 = primData.uvs[index0];
            Vector2f uv1 = primData.uvs[index1];
            Vector2f uv2 = primData.uvs[index2];
            Vector2f uv = (baryCoords[0] * uv0 +
                           baryCoords[1] * uv1 +
                           baryCoords[2] * uv2);

            bool opaque = (*alphaMap)(uv);
            intersects &= opaque;
        }

        if(intersects)
        {
            QuatF q0 = primData.tbnRotations[index0].Normalize();
            QuatF q1 = primData.tbnRotations[index1].Normalize();
            QuatF q2 = primData.tbnRotations[index2].Normalize();
            QuatF tbn = Quat::BarySLerp(q0, q1, q2,
                                        baryCoords[0],
                                        baryCoords[1]);
            // Tangent Space to Local Space Transform
            Vector3 Z_AXIS = ZAxis;
            normal = tbn.Conjugate().ApplyRotation(Z_AXIS);
            // Local Space to World Space Transform
            normal = transform.LocalToWorld(normal);
        }

        // TODO: THIS IS WRONG?
        // Since alpha map can cull particular positions of the primitive
        // pdf is not uniform (thus it is not 1/Area)
        // fix it later since it is not common to lights having alpha mapped primitive

        if(intersects)
        {
            // Approximate the area with the determinant
            float area = TriFunctions::Area(transform, primitiveId, primData);
            pdf = 1.0f / area;
        }
        else pdf = 0.0f;
    }

    __device__ inline
    static float PositionPdfFromHit(// Inputs
                                    const Vector3f&,
                                    const Vector3f&,
                                    const QuatF&,
                                    const GPUTransformI& transform,
                                    //
                                    const PrimitiveId primitiveId,
                                    const TriData& primData)
    {
        // Approximate the area with the determinant
        float area = TriFunctions::Area(transform, primitiveId, primData);
        return 1.0f / area;
    }

    template <class GPUTransform>
    __device__ inline
    static bool IntersectsT(// Output
                            float& newT,
                            TriangleHit& newHit,
                            // I-O
                            const RayReg& rayData,
                            // Input
                            const GPUTransform& transform,
                            const DefaultLeaf& leaf,
                            const TriData& primData)
    {
        // Find the primitive
        float batchIndex;
        GPUFunctions::BinarySearchInBetween(batchIndex, leaf.primitiveId, primData.primOffsets, primData.primBatchCount);
        uint32_t batchIndexInt = static_cast<uint32_t>(batchIndex);

        const bool cullBackface = primData.cullFace[batchIndexInt];

        // Get Position
        uint64_t index0 = primData.indexList[leaf.primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[leaf.primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[leaf.primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        // Do Intersection test on local space
        RayF r = transform.WorldToLocal(rayData.ray);
        //
        float t;
        Vector3 baryCoords;
        bool intersects = r.IntersectsTriangle(baryCoords, t,
                                               position0,
                                               position1,
                                               position2,
                                               cullBackface);
        if(intersects)
        {
            newT = t;
            newHit = Vector2f(baryCoords[0], baryCoords[1]);
        }
        return intersects;
    }

    static constexpr auto& Intersects = IntersectsT<GPUTransformI>;

    __device__ inline
    static bool AlphaTest(// Input
                          const TriangleHit& potentialHit,
                          const DefaultLeaf& leaf,
                          const TriData& primData)
    {
        // Find the primitive
        float batchIndex;
        GPUFunctions::BinarySearchInBetween(batchIndex, leaf.primitiveId, primData.primOffsets, primData.primBatchCount);
        uint32_t batchIndexInt = static_cast<uint32_t>(batchIndex);
        const GPUBitmap* alphaMap = primData.alphaMaps[batchIndexInt];
        // Check if an alpha map does not exist
        // Accept intersection
        if(!alphaMap) return true;

        uint64_t index0 = primData.indexList[leaf.primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[leaf.primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[leaf.primitiveId * 3 + 2];

        Vector3 baryCoords = Vector3f(potentialHit[0],
                                      potentialHit[1],
                                      1.0f - potentialHit[1] - potentialHit[0]);

        Vector2f uv0 = primData.uvs[index0];
        Vector2f uv1 = primData.uvs[index1];
        Vector2f uv2 = primData.uvs[index2];

        Vector2f uv = (baryCoords[0] * uv0 +
                        baryCoords[1] * uv1 +
                        baryCoords[2] * uv2);

        return (*alphaMap)(uv);
    }

    __device__ inline
    static AABB3f AABB(const GPUTransformI& transform,
                       //
                       PrimitiveId primitiveId,
                       const TriData& primData)
    {
        // Get Position
        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        position0 = transform.LocalToWorld(position0);
        position1 = transform.LocalToWorld(position1);
        position2 = transform.LocalToWorld(position2);

        return Triangle::BoundingBox(position0, position1, position2);
    }

    __device__ inline
    static float Area(const GPUTransformI& transform,
                      PrimitiveId primitiveId,
                      const TriData& primData)
    {
        // Get Position
        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        position0 = transform.LocalToWorld(position0);
        position1 = transform.LocalToWorld(position1);
        position2 = transform.LocalToWorld(position2);

        // CCW
        Vector3 vec0 = position1 - position0;
        Vector3 vec1 = position2 - position0;

        return Cross(vec0, vec1).Length() * 0.5f;
    }

    __device__ inline
    static Vector3 Center(const GPUTransformI& transform,
                          PrimitiveId primitiveId, const TriData& primData)
    {
        // Get Position
        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];

        Vector3 position0 = primData.positions[index0];
        Vector3 position1 = primData.positions[index1];
        Vector3 position2 = primData.positions[index2];

        position0 = transform.LocalToWorld(position0);
        position1 = transform.LocalToWorld(position1);
        position2 = transform.LocalToWorld(position2);

        return (position0 * 0.33333f +
                position1 * 0.33333f +
                position2 * 0.33333f);
    }

    __device__ inline
    static void AcquirePositions(// Output
                                 Vector3f positions[3],
                                 // Inputs
                                 PrimitiveId primitiveId,
                                 const TriData& primData)
    {
        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];
        positions[0] = primData.positions[index0];
        positions[1] = primData.positions[index1];
        positions[2] = primData.positions[index2];
    }

    static constexpr auto& Leaf = GenerateDefaultLeaf<TriData>;

    __device__ inline
    static uint32_t Voxelize(uint64_t* gVoxMortonCodes,
                             Vector2us* gVoxelNormals,
                             uint32_t voxIndMaxCount,
                             // Inputs
                             bool onlyCalcSize,
                             PrimitiveId primitiveId,
                             const TriData& primData,
                             const GPUTransformI& transform,
                             // Voxel Inputs
                             const AABB3f& sceneAABB,
                             uint32_t resolutionXYZ)
    {
        // Clang signbit definition is only on std namespace
        // this is a crappy workaround, since this is only a device function
        // but clang gives an error
        #ifndef __CUDA_ARCH__
            using namespace std;
        #endif

        static constexpr Vector3f X_AXIS = XAxis;
        static constexpr Vector3f Y_AXIS = YAxis;

        Vector3f normal;
        Vector3f positions[3];

        uint64_t index0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t index1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t index2 = primData.indexList[primitiveId * 3 + 2];
        positions[0] = primData.positions[index0];
        positions[1] = primData.positions[index1];
        positions[2] = primData.positions[index2];

        // Convert positions to World Space
        positions[0] = transform.LocalToWorld(positions[0]);
        positions[1] = transform.LocalToWorld(positions[1]);
        positions[2] = transform.LocalToWorld(positions[2]);

        //// Find the primitive
        //float batchIndex;
        //GPUFunctions::BinarySearchInBetween(batchIndex, primitiveId,
        //                                    primData.primOffsets, primData.primBatchCount);
        //uint32_t batchIndexInt = static_cast<uint32_t>(batchIndex);
        //const GPUBitmap* alphaMap = primData.alphaMaps[batchIndexInt];

        //// Fetch UV's for potential alpha map check
        //Vector2f uv0 = primData.uvs[index0];
        //Vector2f uv1 = primData.uvs[index1];
        //Vector2f uv2 = primData.uvs[index2];

        // World Space Normal (Will be used to determine best projection plane)
        normal = Triangle::Normal(positions);
        // Find the best projection plane (XY, YZ, XZ)
        int domAxis = normal.Abs().Max();
        bool hasNegSign = signbit(normal[domAxis]);
        float domSign = hasNegSign ? -1.0f : 1.0f;

        // Calculate Spherical UV Coordinates of the normal
        Vector2f sphrCoords = Utility::CartesianToSphericalUnit(normal);
        sphrCoords[0] = (sphrCoords[0] + MathConstants::Pi) * MathConstants::InvPi * 0.5f;
        sphrCoords[1] = sphrCoords[1] * MathConstants::InvPi;
        // Due to numerical error this could slightly exceed [0, 65535]
        // clamp it
        Vector2i sphrUnormInt = Vector2i(static_cast<int32_t>(sphrCoords[0] * 65535.0f),
                                         static_cast<int32_t>(sphrCoords[1] * 65535.0f));
        sphrUnormInt.ClampSelf(0, 65535);
        Vector2us sphrUnorm = Vector2us(sphrUnormInt[0], sphrUnormInt[1]);

        // Look towards to the dominant axis
        QuatF rot;
        switch(domAxis)
        {
            case 0: rot = QuatF(MathConstants::Pi * 0.5f, -domSign * Y_AXIS); break;
            case 1: rot = QuatF(MathConstants::Pi * 0.5f, domSign * X_AXIS); break;
            case 2: rot = (hasNegSign) ? QuatF(MathConstants::Pi, Y_AXIS) : IdentityQuatF; break;
            default: assert(false); return 0;
        }
        // Generate a projection matrix (orthogonal)
        Matrix4x4 proj = TransformGen::Ortogonal(sceneAABB.Min()[0], sceneAABB.Max()[0],
                                                 sceneAABB.Max()[1], sceneAABB.Min()[1],
                                                 sceneAABB.Min()[2], sceneAABB.Max()[2]);

        // Apply Transformations
        Vector3f positionsT[3];
        // First Project
        positionsT[0] = Vector3f(proj * Vector4f(positions[0], 1.0f));
        positionsT[1] = Vector3f(proj * Vector4f(positions[1], 1.0f));
        positionsT[2] = Vector3f(proj * Vector4f(positions[2], 1.0f));
        // Now rotate towards Z axis
        positionsT[0] = rot.ApplyRotation(positionsT[0]);
        positionsT[1] = rot.ApplyRotation(positionsT[1]);
        positionsT[2] = rot.ApplyRotation(positionsT[2]);

        Vector2f positions2D[3];
        positions2D[0] = Vector2f(positionsT[0]);
        positions2D[1] = Vector2f(positionsT[1]);
        positions2D[2] = Vector2f(positionsT[2]);

        // Finally Triangle is on NDC
        // Find AABB then start scan line
        Vector2f aabbMin = Vector2f(FLT_MAX);
        Vector2f aabbMax = Vector2f(-FLT_MAX);
        aabbMin = Vector2f::Min(aabbMin, positions2D[0]);
        aabbMin = Vector2f::Min(aabbMin, positions2D[1]);
        aabbMin = Vector2f::Min(aabbMin, positions2D[2]);

        aabbMax = Vector2f::Max(aabbMax, positions2D[0]);
        aabbMax = Vector2f::Max(aabbMax, positions2D[1]);
        aabbMax = Vector2f::Max(aabbMax, positions2D[2]);

        // Convert to [0, resolution] (pixel space)
        Vector2i xRangeInt(floor((0.5f + 0.5f * aabbMin[0]) * static_cast<float>(resolutionXYZ)),
                           ceil((0.5f + 0.5f * aabbMax[0]) * static_cast<float>(resolutionXYZ)));
        Vector2i yRangeInt(floor((0.5f + 0.5f * aabbMin[1]) * static_cast<float>(resolutionXYZ)),
                           ceil((0.5f + 0.5f * aabbMax[1]) * static_cast<float>(resolutionXYZ)));
        // Clip the range
        xRangeInt.ClampSelf(0, resolutionXYZ);
        yRangeInt.ClampSelf(0, resolutionXYZ);

        // Conservative Rasterization
        // Move all the edges "outwards" at least half a pixel
        // Notice NDC is [-1, 1] pixel size is 2 / resolution
        const float halfPixel = 1.0f / static_cast<float>(resolutionXYZ);
        const float deltaPix = 2.0f * halfPixel;
        // https://developer.nvidia.com/gpugems/gpugems2/part-v-image-oriented-computing/chapter-42-conservative-rasterization
        // This was CG shader code which was optimized
        // with a single cross product you can find the line equation
        // ax + by + c = 0 (planes variable holds a,b,c)
        Vector3f planes[3];
        planes[0] = Cross(Vector3f(positions2D[1] - positions2D[0], 0.0f),
                          Vector3f(positions2D[0], 1.0f));
        planes[1] = Cross(Vector3f(positions2D[2] - positions2D[1], 0.0f),
                          Vector3f(positions2D[1], 1.0f));
        planes[2] = Cross(Vector3f(positions2D[0] - positions2D[2], 0.0f),
                          Vector3f(positions2D[2], 1.0f));
        // Move the planes by the appropriate diagonal
        planes[0][2] -= Vector2f(halfPixel).Dot(Vector2f(planes[0]).Abs());
        planes[1][2] -= Vector2f(halfPixel).Dot(Vector2f(planes[1]).Abs());
        planes[2][2] -= Vector2f(halfPixel).Dot(Vector2f(planes[2]).Abs());
        // Compute the intersection point of the planes.
        // Again this code utilizes cross product to find x,y positions with (w)
        // which was implicitly divided by the rasterizer pipeline
        Vector3f positionsConserv[3];
        positionsConserv[0] = Cross(planes[0], planes[2]);
        positionsConserv[1] = Cross(planes[0], planes[1]);
        positionsConserv[2] = Cross(planes[1], planes[2]);
        // Manually divide "w" (in this case Z) manually
        Vector2f positionsConsv2D[3];
        positionsConsv2D[0] = Vector2f(positionsConserv[0]) / positionsConserv[0][2];
        positionsConsv2D[1] = Vector2f(positionsConserv[1]) / positionsConserv[1][2];
        positionsConsv2D[2] = Vector2f(positionsConserv[2]) / positionsConserv[2][2];

        // Generate Edges (for Cramer's Rule)
        // & iteration constants
        // Conservative Edges
        const Vector2f eCons0 = positionsConsv2D[1] - positionsConsv2D[0];
        const Vector2f eCons1 = positionsConsv2D[2] - positionsConsv2D[0];
        const float denomCons = 1.0f / (eCons0[0] * eCons1[1] - eCons1[0] * eCons0[1]);
        // Actual Edges
        const Vector2f e0 = positions2D[1] - positions2D[0];
        const Vector2f e1 = positions2D[2] - positions2D[0];
        const float denom = 1.0f / (e0[0] * e1[1] - e1[0] * e0[1]);
        // Scan Line Rasterization
        uint32_t writeIndex = 0;
        for(int y = yRangeInt[0]; y < yRangeInt[1]; y++)
        for(int x = xRangeInt[0]; x < xRangeInt[1]; x++)
        {
            // Gen Point (+0.5 for pixel middle)
            Vector2f pos = Vector2f((static_cast<float>(x) + 0.5f) * deltaPix - 1.0f,
                                    (static_cast<float>(y) + 0.5f) * deltaPix - 1.0f);

            // Cramer's Rule
            Vector2f eCons2 = pos - positionsConsv2D[0];
            float b = (eCons2[0] * eCons1[1] - eCons1[0] * eCons2[1]) * denomCons;
            float c = (eCons0[0] * eCons2[1] - eCons2[0] * eCons0[1]) * denomCons;
            float a = 1.0f - b - c;
            // If barycentrics are in range
            if(b >= 0.0f && b <= 1.0f &&
               c >= 0.0f && c <= 1.0f &&
               a >= 0.0f && a <= 1.0f)
            {
                // Find the Actual Bary Coords here
                // Cramer's Rule
                Vector2f e2 = pos - positions2D[0];
                float actualB = (e2[0] * e1[1] - e1[0] * e2[1]) * denom;
                float actualC = (e0[0] * e2[1] - e2[0] * e0[1]) * denom;
                float actualA = 1.0f - actualB - actualC;

                // TODO:
                // This code below is not proper. You need to multi-sample
                // or utilize proper mip-map level fetching
                // METUray does not hold mipmapped alpha maps
                // (in this case it should be conservative)

                // Check alpha map if available
                //bool isTransparent = false;
                //if(alphaMap)
                //{
                //    Vector2f uv = (actualA * uv0 +
                //                   actualB * uv1 +
                //                   actualC * uv2);
                //    isTransparent = !((*alphaMap)(uv));
                //}
                //if(isTransparent) continue;

                // Bary's match, pixel is inside the triangle
                if(!onlyCalcSize)
                {
                    Vector3f voxelPos = (positions[0] * actualA +
                                         positions[1] * actualB +
                                         positions[2] * actualC);

                    Vector3f voxelIndexF = ((voxelPos - sceneAABB.Min()) / sceneAABB.Span());
                    voxelIndexF *= static_cast<float>(resolutionXYZ);
                    Vector3ui voxelIndex = Vector3ui(static_cast<uint32_t>(voxelIndexF[0]),
                                                     static_cast<uint32_t>(voxelIndexF[1]),
                                                     static_cast<uint32_t>(voxelIndexF[2]));

                    // TODO: This sometimes happen but it shouldn't??
                    // Clamp the Voxel due to numerical errors
                    voxelIndex.ClampSelf(0, resolutionXYZ - 1);

                    uint64_t voxelIndexMorton = MortonCode::Compose3D<uint64_t>(voxelIndex);
                    // Write the found voxel
                    assert(writeIndex < voxIndMaxCount);
                    if(writeIndex < voxIndMaxCount)
                    {
                        gVoxMortonCodes[writeIndex] = voxelIndexMorton;
                        gVoxelNormals[writeIndex] = sphrUnorm;
                    }
                };
                writeIndex++;
            }
        }
        return writeIndex;
    }
};

class GPUPrimitiveTriangle;

struct TriangleSurfaceGenerator
{
    __device__ inline
    static BasicSurface GenBasicSurface(const TriangleHit& baryCoords,
                                        const GPUTransformI& transform,
                                        //
                                        const Vector3f& rayDir,
                                        //
                                        PrimitiveId primitiveId,
                                        const TriData& primData)
    {
        // Check if the prim is two-sided
        float batchIndex;
        GPUFunctions::BinarySearchInBetween(batchIndex, primitiveId, primData.primOffsets, primData.primBatchCount);
        uint32_t batchIndexInt = static_cast<uint32_t>(batchIndex);
        bool twoSided = !primData.cullFace[batchIndexInt];

        float a = baryCoords[0];
        float b = baryCoords[1];
        float c = 1.0f - a -  b;

        uint64_t i0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t i1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t i2 = primData.indexList[primitiveId * 3 + 2];

        // Tangent Space Rotation Query
        QuatF q0 = primData.tbnRotations[i0];
        QuatF q1 = primData.tbnRotations[i1];
        QuatF q2 = primData.tbnRotations[i2];
        QuatF tbn = Quat::BarySLerp(q0, q1, q2,
                                    a, b);
        tbn.NormalizeSelf();
        QuatF worldToTangent = tbn * transform.ToLocalRotation();

        // Position Query
        Vector3f p0 = primData.positions[i0];
        Vector3f p1 = primData.positions[i1];
        Vector3f p2 = primData.positions[i2];
        Vector3f pos = (p0 * a + p1 * b + p2 * c);
        // Position should be in world space
        pos = transform.LocalToWorld(pos);

        // Calculate Geometric Normal
        const Vector3f positions[3] = {p0, p1, p2};
        Vector3f geoNormal = Triangle::Normal(positions);
        // This also needs to be in world space
        geoNormal = transform.LocalToWorld(geoNormal, true).Normalize();

        // If The requested primitive is two sided
        // Flip the surface definitions (normal, geometric normal)
        bool backSide = twoSided && (geoNormal.Dot(rayDir) > 0.0f);
        if(backSide)
        {
            geoNormal = -geoNormal;
            // Change the worldToTangent rotation so that Z is on opposite direction
            // TODO: here flipping Z would change the handedness of the
            // coordinate system
            // Just adding the 180degree rotation with the tangent axis
            // to the end which should be fine I guess?
            static constexpr QuatF TANGENT_ROT = QuatF(0, 1, 0, 0);
            worldToTangent = TANGENT_ROT * worldToTangent;
        }

        //// Calculate the curvature offset
        //// https://computergraphics.stackexchange.com/questions/1718/what-is-the-simplest-way-to-compute-principal-curvature-for-a-mesh-triangle
        //// Calculate curvature for each three edges then use the max
        //auto CalculateCurvatureDistance = [&](const Vector3f& p0, const Vector3f& p1,
        //                                      const Vector3f& n0, const Vector3f& n1,
        //                                      const Vector3f& nPatch) -> float
        //{
        //    Vector3f pVec = (p1 - p0);
        //    Vector3f nVec = (n1 - n0);

        //    float curvature = pVec.Dot(nVec) / pVec.LengthSqr();
        //    // For back faced surfaces curvature is inverse
        //    if(backSide) curvature = -curvature;
        //    // Just don't generate a negative offset, early terminate
        //    if(curvature <= MathConstants::Epsilon) return 0.0f;
        //    // Assume circle
        //    float radius = (1.0f / curvature);
        //    // Geometrically determine the distance between this square and
        //    // the chord (which is current processed edge)
        //    // alpha = half angle between normals
        //    // beta = angle between center normal and the interpolating normal
        //    //
        //    //   N0   Ncenter    N1
        //    //     ^     ^     ^
        //    //      \____|____/
        //    //       \   |   /
        //    //        \  |  /
        //    //         \ | /
        //    //          \|/
        //    // angle between = N0-N1 is alpha,
        //    // (thus angle between Ncenter-N0 is alpha/2)
        //    // angle between nPatch and nCenter is beta

        //    Vector3f nCenter = (n0 + n1).Normalize();
        //    float cosAlphaHalf = n0.Dot(nCenter);
        //    assert(fabsf(n1.Dot(nCenter) - cosAlphaHalf) < MathConstants::VeryLargeEpsilon);
        //    float cosBeta = nPatch.Dot(nCenter);
        //    float dist = radius * (cosBeta - cosAlphaHalf);
        //    // Numerical inaccuracies may put the angle beta outside of alpha
        //    // which means the angles are extremely close just clamp it to zero
        //    dist = fmax(dist, 0.0f);

        //    assert(dist >= 0);
        //    return dist;
        //};

        //Vector3f nPatch = GPUSurface::NormalToSpace(tbn);
        //Vector3f n0 = GPUSurface::NormalToSpace(q0);
        //Vector3f n1 = GPUSurface::NormalToSpace(q1);
        //Vector3f n2 = GPUSurface::NormalToSpace(q2);

        //auto Project = [](const Vector3f& a, const Vector3f& p0,
        //                  const Vector3f& p1)
        //{
        //    Vector3f planeNormal = Cross(p0, p1);
        //    Vector3f pNorm = planeNormal.Normalize();
        //    if(planeNormal.LengthSqr() < MathConstants::VeryLargeEpsilon)
        //    {
        //        // Just return any of the two plane vectors
        //        return p0;
        //    }
        //    // Project to the plane
        //    return  a - pNorm.Dot(a) * pNorm;
        //};

        //float dist0 = CalculateCurvatureDistance(p0, p1,
        //                                         n0, n1,
        //                                         Project(nPatch, n0, n1));
        //float dist1 = CalculateCurvatureDistance(p1, p2,
        //                                         n1, n2,
        //                                         Project(nPatch, n1, n2));
        //float dist2 = CalculateCurvatureDistance(p2, p0,
        //                                         n2, n0,
        //                                         Project(nPatch, n2, n0));
        //float curvatureOffset = fmax(fmax(dist0, dist1), dist2);


        return BasicSurface{pos, worldToTangent, geoNormal, backSide, 0.0f};
    }

    __device__ inline
    static BarySurface GenBarySurface(const TriangleHit& baryCoords,
                                      const GPUTransformI&,
                                      //
                                      const Vector3f&,
                                      //
                                      PrimitiveId,
                                      const TriData&)
    {
        float c = 1.0f - baryCoords[0] - baryCoords[1];
        return BarySurface{Vector3(baryCoords[0], baryCoords[1], c)};
    }

    __device__ inline
    static UVSurface GenUVSurface(const TriangleHit& baryCoords,
                                  const GPUTransformI& transform,
                                  //
                                  const Vector3f& rayDir,
                                  //
                                  PrimitiveId primitiveId,
                                  const TriData& primData)
    {
        BasicSurface bs = GenBasicSurface(baryCoords, transform,
                                          rayDir, primitiveId,
                                          primData);

        float c = 1 - baryCoords[0] - baryCoords[1];

        uint64_t i0 = primData.indexList[primitiveId * 3 + 0];
        uint64_t i1 = primData.indexList[primitiveId * 3 + 1];
        uint64_t i2 = primData.indexList[primitiveId * 3 + 2];

        Vector2 uv0 = primData.uvs[i0];
        Vector2 uv1 = primData.uvs[i1];
        Vector2 uv2 = primData.uvs[i2];

        Vector2 uv = (uv0 * baryCoords[0] +
                      uv1 * baryCoords[1] +
                      uv2 * c);

        return UVSurface
        {
            bs.worldPosition, bs.worldToTangent,
            uv, bs.worldGeoNormal, bs.backSide,
            bs.curvatureOffset
        };
    }

    template <class Surface, SurfaceFunc<Surface, TriangleHit, TriData> SF>
    struct SurfaceFunctionType
    {
        using type = Surface;
        static constexpr auto SurfaceGeneratorFunction = SF;
    };

    static constexpr auto GeneratorFunctionList =
        std::make_tuple(SurfaceFunctionType<EmptySurface, DefaultGenEmptySurface<TriangleHit, TriData>>{},
                        SurfaceFunctionType<BasicSurface, GenBasicSurface>{},
                        SurfaceFunctionType<BarySurface, GenBarySurface>{},
                        SurfaceFunctionType<UVSurface, GenUVSurface>{});

    template<class Surface>
    static constexpr SurfaceFunc<Surface, TriangleHit, TriData> GetSurfaceFunction()
    {
        using namespace PrimitiveSurfaceFind;
        return LoopAndFindType<Surface, SurfaceFunc<Surface, TriangleHit, TriData>,
                               decltype(GeneratorFunctionList)>(std::move(GeneratorFunctionList));
    }
};

class GPUPrimitiveTriangle final
    : public GPUPrimitiveGroup<TriangleHit, TriData, DefaultLeaf,
                               TriangleSurfaceGenerator, TriFunctions,
                               PrimTransformType::CONSTANT_LOCAL_TRANSFORM,
                               3>
{
    public:
        static constexpr const char*            TypeName() { return "Triangle"; }

        static constexpr PrimitiveDataLayout    POS_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    UV_LAYOUT = PrimitiveDataLayout::FLOAT_2;
        static constexpr PrimitiveDataLayout    NORMAL_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    TANGENT_LAYOUT = PrimitiveDataLayout::FLOAT_3;
        static constexpr PrimitiveDataLayout    INDEX_LAYOUT = PrimitiveDataLayout::UINT64_1;

        static constexpr const char*            CULL_FLAG_NAME = "cullFace";
        static constexpr const char*            ALPHA_MAP_NAME = "alphaMap";

        using LoadedBitmapIndices = std::map<std::pair<uint32_t, TextureChannelType>, uint32_t>;

    private:
        DeviceMemory                            memory;
        // List of ranges for each batch
        uint64_t                                totalPrimitiveCount;
        uint64_t                                totalDataCount;
        // CPU Allocation of Bitmaps
        LoadedBitmapIndices                     loadedBitmaps;
        CPUBitmapGroup                          bitmaps;
        // Misc Data
        std::map<uint32_t, Vector2ul>           batchRanges;
        std::map<uint32_t, Vector2ul>           batchDataRanges;
        std::map<uint32_t, AABB3>               batchAABBs;
        std::map<uint32_t, bool>                batchAlphaMapFlag;
        std::map<uint32_t, bool>                batchBackFaceCullFlag;

    protected:
    public:
        // Constructors & Destructor
                                                GPUPrimitiveTriangle();
                                                ~GPUPrimitiveTriangle() = default;

        // Interface
        // Primitive type is used for delegating scene info to this class
        const char*                             Type() const override;
        // Allocates and Generates Data
        SceneError                              InitializeGroup(const NodeListing& surfaceDataNodes, double time,
                                                                const SurfaceLoaderGeneratorI&,
                                                                const TextureNodeMap& textureNodes,
                                                                const std::string& scenePath) override;
        SceneError                              ChangeTime(const NodeListing& surfaceDataNodes, double time,
                                                           const SurfaceLoaderGeneratorI&,
                                                           const std::string& scenePath) override;
        // Access primitive range from Id
        Vector2ul                               PrimitiveBatchRange(uint32_t surfaceDataId) const override;
        AABB3                                   PrimitiveBatchAABB(uint32_t surfaceDataId) const override;
        bool                                    PrimitiveBatchHasAlphaMap(uint32_t surfaceDataId) const override;
        bool                                    PrimitiveBatchBackFaceCulled(uint32_t surfaceDataId) const override;
        // Query
        // How many primitives are available on this class
        // This includes the indexed primitive count
        uint64_t                                TotalPrimitiveCount() const override;
        // Total primitive count but not indexed
        uint64_t                                TotalDataCount() const override;
        // Primitive Transform Info for accelerator
        bool                                    IsTriangle() const override;
        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        bool                                    CanGenerateData(const std::string& s) const override;
};

static_assert(IsTracerClass<GPUPrimitiveTriangle>::value,
              "GPUPrimitiveTriangle is not a Tracer Class.");