#pragma once

#include "GPUCameraP.cuh"
#include "GPUTransformI.h"
#include "TypeTraits.h"
#include "GPUCameraPixel.cuh"
#include "MangledNames.h"

class GPUCameraPinhole final : public GPUCameraI
{
    private:
    protected:
        Vector3                 position;
        Vector3                 right;
        Vector3                 up;
        Vector3                 gazeDir;
        Vector3                 bottomLeft;
        Vector2                 planeSize;
        Vector2                 nearFar;
        Vector2                 fov;

    protected:
    public:
        // Constructors & Destructor
        __device__          GPUCameraPinhole(const Vector3& position,
                                             const Vector3& gaze,
                                             const Vector3& up,
                                             const Vector2& nearFar,
                                             const Vector2& fov,
                                             // Base Class Related
                                             uint16_t mediumId, HitKey hk,
                                             const GPUTransformI& gTrans);
                            ~GPUCameraPinhole() = default;

        // Interface
        __device__ void     Sample(// Output
                                   float& distance,
                                   Vector3& direction,
                                   float& pdf,
                                   Vector2f& localCoords,
                                   // Input
                                   const Vector3& position,
                                   // I-O
                                   RNGeneratorGPUI&) const override;

        __device__ void     GenerateRay(// Output
                                        RayReg&,
                                        Vector2f& localCoords,
                                        // Input
                                        const Vector2i& sampleId,
                                        const Vector2i& sampleMax,
                                        // I-O
                                        RNGeneratorGPUI&,
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

class CPUCameraGroupPinhole final : public CPUCameraGroupP<GPUCameraPinhole>
{
    public:
        TYPENAME_DEF(CameraGroup, "Pinhole");

        // Node Names
        static constexpr const char* POSITION_NAME  = "position";
        static constexpr const char* GAZE_NAME      = "gaze";
        static constexpr const char* UP_NAME        = "up";
        static constexpr const char* PLANES_NAME    = "planes";
        static constexpr const char* FOV_NAME       = "fov";

        struct Data
        {
            // Sample Ready Parameters
            // All of which is world space
            Vector3                 position;
            Vector3                 gaze;
            Vector3                 up;
            Vector2                 nearFar;
            Vector2                 fov;
        };

        using Base = CPUCameraGroupP<GPUCameraPinhole>;

    private:
        std::vector<Data>               hCameraData;

    protected:
    public:
        // Constructors & Destructor
                                        CPUCameraGroupPinhole(const GPUPrimitiveGroupI*);
                                        ~CPUCameraGroupPinhole() = default;

        // Interface
        const char*                     Type() const override;
        SceneError					    InitializeGroup(const EndpointGroupDataList& cameraNodes,
                                                        const TextureNodeMap& textures,
                                                        const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                        const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                        uint32_t batchId, double time,
                                                        const std::string& scenePath) override;
        SceneError					    ChangeTime(const NodeListing& lightNodes, double time,
                                                   const std::string& scenePath) override;
        TracerError					    ConstructEndpoints(const GPUTransformI**,
                                                           const AABB3f&,
                                                           const CudaSystem&) override;

        size_t						    UsedCPUMemory() const override;
};

__device__
inline GPUCameraPinhole::GPUCameraPinhole(const Vector3& pos,
                                          const Vector3& gz,
                                          const Vector3& upp,
                                          const Vector2& nF,
                                          const Vector2& fov,
                                          // Base Class Related
                                          uint16_t mediumId, HitKey hk,
                                          const GPUTransformI& gTrans)
    : GPUCameraI(mediumId, hk, gTrans)
    , position(gTrans.LocalToWorld(pos))
    , up(gTrans.LocalToWorld(upp))
    , nearFar(nF)
    , fov(fov)
{
    Vector3 gazePoint = gTrans.LocalToWorld(gz);
    float nearPlane = nearFar[0];
    float farPLane = nearFar[1];

    // Find world space window sizes
    float widthHalf = tanf(fov[0] * 0.5f) * nearPlane;
    float heightHalf = tanf(fov[1] * 0.5f) * nearPlane;

    // Camera Vector Correction
    gazeDir = gazePoint - position;
    right = Cross(gazeDir, up).Normalize();
    up = Cross(right, gazeDir).Normalize();
    gazeDir = Cross(up, right).Normalize();

    // Camera parameters
    bottomLeft = (position
                  - right * widthHalf
                  - up * heightHalf
                  + gazeDir * nearPlane);

    planeSize = Vector2(widthHalf, heightHalf) * 2.0f;
}

__device__
inline void GPUCameraPinhole::Sample(// Output
                                     float& distance,
                                     Vector3& direction,
                                     float& pdf,
                                     Vector2f& localCoords,
                                     // Input
                                     const Vector3& sampleLoc,
                                     // I-O
                                     RNGeneratorGPUI&) const
{
    // One
    direction = sampleLoc - position;
    distance = direction.Length();
    direction.NormalizeSelf();
    pdf = 1.0f;

    // Generate image space (local) coords
    // "bottomLeft" plane is "near" distance away from the position
    float nearPlane = nearFar[0];
    float cosAlpha = gazeDir.Dot(direction);
    float planeDist = nearFar[0] / cosAlpha;

    // Adjust length of the direction
    Vector3f dirNear = direction * planeDist;

    float deltaX = dirNear.Dot(right);
    float deltaY = dirNear.Dot(right);

    Vector2f planeCenter = planeSize * 0.5f;
    Vector2f planeCoord = planeCenter - Vector2f(deltaX, deltaY);
    localCoords = planeCoord / planeSize;
}

__device__
inline void GPUCameraPinhole::GenerateRay(// Output
                                          RayReg& ray,
                                          Vector2f& localCoords,
                                          // Input
                                          const Vector2i& sampleId,
                                          const Vector2i& sampleMax,
                                          // I-O
                                          RNGeneratorGPUI& rng,
                                          // Options
                                          bool antiAliasOn) const
{
    // DX DY from stratified sample
    Vector2 delta = Vector2(planeSize[0] / static_cast<float>(sampleMax[0]),
                            planeSize[1] / static_cast<float>(sampleMax[1]));

    // Create random location over sample rectangle
    Vector2 randomOffset = (antiAliasOn)
                            ? Vector2(rng.Uniform(), rng.Uniform())
                            : Vector2(0.5f);

    Vector2 sampleDistance = Vector2(static_cast<float>(sampleId[0]),
                                     static_cast<float>(sampleId[1])) * delta;
    sampleDistance += (randomOffset * delta);
    Vector3 samplePoint = bottomLeft + ((sampleDistance[0] * right) +
                                        (sampleDistance[1] * up));
    Vector3 rayDir = (samplePoint - position).Normalize();

    // Local Coords
    localCoords = sampleDistance / planeSize;

    // Initialize Ray
    ray.ray = RayF(rayDir, position);
    ray.tMin = nearFar[0];
    ray.tMax = nearFar[1];
}

__device__
inline float GPUCameraPinhole::Pdf(const Vector3& worldDir,
                                   const Vector3& worldPos) const
{
    return 0.0f;
}

__device__
inline float GPUCameraPinhole::Pdf(float distance,
                               const Vector3& hitPosition,
                               const Vector3& direction,
                               const QuatF& tbnRotation) const
{
    return 0.0f;
}

__device__
inline uint32_t GPUCameraPinhole::FindPixelId(const RayReg& r,
                                              const Vector2i& resolution) const
{
    Vector3f normal = Cross(up, right);
    //float ratio = normal.Dot(r.getDirection());
    //// Only front facing intersections are considered
    //if(ratio > 0) return UINT32_MAX;
    //
    //float t = (position - r.getPosition()).Dot(normal) / ratio;
    //// Intersection is behind
    //if(t < 0) return UINT32_MAX;

    Vector3 planePoint = r.ray.AdvancedPos(r.tMax);
    Vector3 p = planePoint - position;

    Matrix3x3 invRot(right[0], right[1], right[2],
                     up[0], up[1], up[2],
                     normal[0], normal[1], normal[2]);
    Vector3 localP = invRot * p;
    // Convert to Pixelated System
    Vector2i coords = Vector2i((Vector2(localP) / planeSize).Floor());

    uint32_t pixelId = coords[1] * resolution[0] + coords[0];
    return pixelId;
}

__device__
inline bool GPUCameraPinhole::CanBeSampled() const
{
    return false;
}

__device__
inline Matrix4x4 GPUCameraPinhole::VPMatrix() const
{
    Matrix4x4 p = TransformGen::Perspective(fov[0], fov[0] / fov[1],
                                            nearFar[0], nearFar[1]);
    Matrix3x3 rotMatrix;
    TransformGen::Space(rotMatrix, right, up, Cross(right, up));
    return p * ToMatrix4x4(rotMatrix);
}

__device__
inline Vector2f GPUCameraPinhole::NearFar() const
{
    return nearFar;
}

__device__
inline VisorTransform GPUCameraPinhole::GenVisorTransform() const
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
inline void GPUCameraPinhole::SwapTransform(const VisorTransform& t)
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
inline GPUCameraPixel GPUCameraPinhole::GeneratePixelCamera(const Vector2i& pixelId,
                                                            const Vector2i& resolution) const
{
    // DX DY from stratified sample
    Vector2 delta = Vector2(planeSize[0] / static_cast<float>(resolution[0]),
                            planeSize[1] / static_cast<float>(resolution[1]));

    Vector2 pixelDistance = Vector2(static_cast<float>(pixelId[0]),
                                    static_cast<float>(pixelId[1])) * delta;
    Vector3 pixelBottomLeft = bottomLeft + ((pixelDistance[0] * right) +
                                            (pixelDistance[1] * up));

    return GPUCameraPixel(position,
                          right,
                          up,
                          pixelBottomLeft,
                          delta,
                          nearFar,
                          pixelId,
                          resolution,
                          mediumIndex,
                          workKey,
                          gTransform);
}

inline CPUCameraGroupPinhole::CPUCameraGroupPinhole(const GPUPrimitiveGroupI* pg)
    : Base(*pg)
{}

inline const char* CPUCameraGroupPinhole::Type() const
{
    return TypeName();
}

inline size_t CPUCameraGroupPinhole::UsedCPUMemory() const
{
    size_t totalSize = (Base::UsedCPUMemory() +
                        sizeof(Data) * hCameraData.size());
    return totalSize;
}

static_assert(IsTracerClass<CPUCameraGroupPinhole>::value,
              "CPUCameraPinhole is not a tracer class");