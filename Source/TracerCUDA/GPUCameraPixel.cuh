#pragma once

#include "GPUCameraI.h"
#include "RayLib/VisorTransform.h"

class GPUCameraPixel final : public GPUCameraI
{
    private:
        // Pixel Location Related
        Vector3                 position;
        Vector3                 right;
        Vector3                 up;
        Vector3                 bottomLeft;
        Vector2                 planeSize;
        Vector2                 nearFar;

        Vector2i                pixelId;
        Vector2i                resolution;

    protected:
    public:
        // Constructors & Destructor
        __device__          GPUCameraPixel(const Vector3& position,
                                           const Vector3& right,
                                           const Vector3& up,
                                           const Vector3& bottomLeft,
                                           const Vector2& planeSize,
                                           const Vector2& nearFar,
                                           const Vector2i& pixelId,
                                           const Vector2i& resolution,
                                           // Base Class Related
                                           uint16_t mediumId, HitKey hk,
                                           const GPUTransformI& gTrans);
                            ~GPUCameraPixel() = default;

        // Interface
        __device__ void     Sample(// Output
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RandomGPU&) const override;

        __device__ void     GenerateRay(// Output
                                        RayReg&,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU&,
                                        // Options
                                        bool antiAliasOn) const override;
        __device__ float    Pdf(const Vector3& direction,
                                const Vector3& position) const override;
        __device__ float    Pdf(float distance,
                                const Vector3& hitPosition,
                                const Vector3& direction,
                                const QuatF& tbnRotation) const override;


        __device__ uint32_t         FindPixelId(const RayReg& r,
                                               const Vector2i& resolution) const override;

        __device__ bool             CanBeSampled() const override;

        __device__ Matrix4x4        VPMatrix() const override;
        __device__ Vector2f         NearFar() const override;

        __device__ VisorTransform   GenVisorTransform() const override;
        __device__ void             SwapTransform(const VisorTransform&) override;

        __device__ GPUCameraPixel   GeneratePixelCamera(const Vector2i& pixelId,
                                                        const Vector2i& resolution) const override;
};

__device__
inline GPUCameraPixel::GPUCameraPixel(const Vector3& position,
                                      const Vector3& right,
                                      const Vector3& up,
                                      const Vector3& bottomLeft,
                                      const Vector2& planeSize,
                                      const Vector2& nearFar,
                                      const Vector2i& pixelId,
                                      const Vector2i& resolution,
                                      // Base Class Related
                                      uint16_t mediumId, HitKey hk,
                                      const GPUTransformI& gTrans)
    : GPUCameraI(mediumId, hk, gTrans)
    , position(position)
    , right(right)
    , up(up)
    , bottomLeft(bottomLeft)
    , planeSize(planeSize)
    , nearFar(nearFar)
    , pixelId(pixelId)
    , resolution(resolution)
{}

__device__
inline void GPUCameraPixel::Sample(// Output
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   // Input
                                   const Vector3& sampleLoc,
                                   // I-O
                                   RandomGPU& rng) const
{
    // One
    direction = sampleLoc - position;
    distance = direction.Length();
    direction.NormalizeSelf();
    pdf = 1.0f;
}

__device__
inline void GPUCameraPixel::GenerateRay(// Output
                                        RayReg& ray,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RandomGPU& rng,
                                        // Options
                                        bool antiAliasOn) const
{
    // DX DY from stratfied sample
    Vector2 delta = Vector2(planeSize[0] / static_cast<float>(sampleMax[0]),
                            planeSize[1] / static_cast<float>(sampleMax[1]));

    // Create random location over sample rectangle
    Vector2 randomOffset = (antiAliasOn)
                            ? Vector2(GPUDistribution::Uniform<float>(rng),
                                      GPUDistribution::Uniform<float>(rng))
                            : Vector2(0.5f);

    Vector2 sampleDistance = Vector2(static_cast<float>(sampleId[0]),
                                     static_cast<float>(sampleId[1])) * delta;
    sampleDistance += (randomOffset * delta);
    Vector3 samplePoint = bottomLeft + ((sampleDistance[0] * right) +
                                        (sampleDistance[1] * up));
    Vector3 rayDir = (samplePoint - position).Normalize();

    // Initialize Ray
    ray.ray = RayF(rayDir, position);
    ray.tMin = nearFar[0];
    ray.tMax = nearFar[1];
}

__device__
inline float GPUCameraPixel::Pdf(const Vector3& worldDir,
                                 const Vector3& worldPos) const
{
    return 0.0f;
}

__device__
inline float GPUCameraPixel::Pdf(float distance,
                                 const Vector3& hitPosition,
                                 const Vector3& direction,
                                 const QuatF& tbnRotation) const
{
    return 0.0f;
}

__device__
inline uint32_t GPUCameraPixel::FindPixelId(const RayReg& r,
                                            const Vector2i& res) const
{
   Vector3f normal = Cross(up, right);
   Vector3 planePoint = r.ray.AdvancedPos(r.tMax);
   Vector3 p = planePoint - position;

    Matrix3x3 invRot(right[0], right[1], right[2],
                     up[0], up[1], up[2],
                     normal[0], normal[1], normal[2]);
    Vector3 localP = invRot * p;
    // Convert to Pixelated System
    Vector2i coords = Vector2i((Vector2(localP) / planeSize).Floor());

    uint32_t pixelId = coords[1] * res[0] + coords[0];
    return pixelId;
}

inline __device__ bool GPUCameraPixel::CanBeSampled() const
{
    return false;
}

__device__
inline Matrix4x4 GPUCameraPixel::VPMatrix() const
{
    Vector3 blDir = (bottomLeft - position).Normalize();
    Vector3 bottomRight = bottomLeft + planeSize[0] * right;
    Vector3 brDir = (bottomRight - position).Normalize();
    float cosFovX = brDir.Dot(blDir);
    float fovX = acos(cosFovX);

    Matrix4x4 p = TransformGen::Perspective(fovX, 1.0f, nearFar[0], nearFar[1]);
    Matrix3x3 rotMatrix;
    TransformGen::Space(rotMatrix, right, up, Cross(right, up));
    return p * ToMatrix4x4(rotMatrix);
}

__device__
inline Vector2f GPUCameraPixel::NearFar() const
{
    return nearFar;
}

__device__
inline VisorTransform GPUCameraPixel::GenVisorTransform() const
{
    Vector3 dir = Cross(up, right).Normalize();
    return VisorTransform
    {
        position,
        position + dir,
        up
    };
}

__device__
inline void GPUCameraPixel::SwapTransform(const VisorTransform& t)
{
    position = t.position;
    up = t.up;
    Vector3 gazePoint = t.gazePoint;

    // Camera Vector Correction
    Vector3 gazeDir = gazePoint - position;
    right = Cross(gazeDir, up).Normalize();
    up = Cross(right, gazeDir).Normalize();
    gazeDir = Cross(up, right).Normalize();

    // Camera parameters
    float widthHalf = planeSize[0] * 0.5f;
    float heightHalf = planeSize[1] * 0.5f;

    bottomLeft = (position
                  - right * widthHalf
                  - up * heightHalf
                  + gazeDir * nearFar[0]);
}

__device__
inline GPUCameraPixel GPUCameraPixel::GeneratePixelCamera(const Vector2i& pId,
                                                          const Vector2i& res) const
{
    // DX DY from stratfied sample
    Vector2 delta = Vector2(planeSize[0] / static_cast<float>(res[0]),
                            planeSize[1] / static_cast<float>(res[1]));

    Vector2 pixelDistance = Vector2(static_cast<float>(pId[0]),
                                    static_cast<float>(pId[1])) * delta;
    Vector3 pixelBottomLeft = bottomLeft + ((pixelDistance[0] * right) +
                                            (pixelDistance[1] * up));

    return GPUCameraPixel(position,
                          right,
                          up,
                          pixelBottomLeft,
                          delta,
                          nearFar,
                          pId,
                          res,
                          mediumIndex,
                          workKey,
                          gTransform);
}

__global__
void KCConstructSingleGPUCameraPixel(GPUCameraPixel* gCameraLocations,
                                     //
                                     const GPUCameraI& baseCam,
                                     Vector2i pixelIndex,
                                     Vector2i resolution);