#pragma once

#include "GPUCameraP.cuh"
#include "GPUTransformI.h"
#include "TypeTraits.h"
#include "GPUCameraPixel.cuh"
#include "MangledNames.h"

#include "RayLib/CoordinateConversion.h"

class GPUCameraSpherical final : public GPUCameraI
{
    private:
        float                   pixelRatio;
        Vector3                 position;
        Vector3                 direction;
        Vector3                 right;
        Vector3                 up;
        Vector2                 nearFar;

    protected:
    public:
        // Constructors & Destructor
        __device__          GPUCameraSpherical(float pixelRatio,
                                               const Vector3& position,
                                               const Vector3& direction,
                                               const Vector3& up,
                                               const Vector2& nearFar,
                                               // Base Class Related
                                               uint16_t mediumId, HitKey hk,
                                               const GPUTransformI& gTrans);
                            ~GPUCameraSpherical() = default;

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

class CPUCameraGroupSpherical final : public CPUCameraGroupP<GPUCameraSpherical>
{
    public:
        TYPENAME_DEF(CameraGroup, "Spherical");

        // Node Names
        static constexpr const char* POSITION_NAME  = "position";
        static constexpr const char* DIR_NAME       = "direction";
        static constexpr const char* UP_NAME        = "up";
        static constexpr const char* PLANES_NAME    = "planes";
        static constexpr const char* PIX_RATIO_NAME = "pixelRatio";

        struct Data
        {
            // Sample Ready Parameters
            // All of which is world space
            float                   pixelRatio;
            Vector3                 position;
            Vector3                 direction;
            Vector3                 up;
            Vector2                 nearFar;
        };

        using Base = CPUCameraGroupP<GPUCameraSpherical>;

    private:
        std::vector<Data>               hCameraData;

    protected:
    public:
        // Constructors & Destructor
                                        CPUCameraGroupSpherical(const GPUPrimitiveGroupI*);
                                        ~CPUCameraGroupSpherical() = default;

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
inline GPUCameraSpherical::GPUCameraSpherical(float pixelRatio,
                                              const Vector3& pos,
                                              const Vector3& dir,
                                              const Vector3& upp,
                                              const Vector2& nearFar,
                                              // Base Class Related
                                              uint16_t mediumId, HitKey hk,
                                              const GPUTransformI& gTrans)
    : GPUCameraI(mediumId, hk, gTrans)
    , pixelRatio(pixelRatio)
    , position(gTrans.LocalToWorld(pos))
    , direction(gTrans.LocalToWorld(dir))
    , up(gTrans.LocalToWorld(upp))
    , nearFar(nearFar)
{
    // Camera Vector Correction
    right = Cross(direction, up).Normalize();
    up = Cross(right, direction).Normalize();
    direction = Cross(up, right).Normalize();
}

__device__
inline void GPUCameraSpherical::Sample(// Output
                                       float& distance,
                                       Vector3& dirOut,
                                       float& pdf,
                                       Vector2f& localCoords,
                                       // Input
                                       const Vector3& sampleLoc,
                                       // I-O
                                       RNGeneratorGPUI&) const
{
    // One
    dirOut = sampleLoc - position;
    distance = dirOut.Length();
    dirOut.NormalizeSelf();
    pdf = 1.0f;

    Vector3 dirZUp = Vector3(dirOut[2], dirOut[0], dirOut[1]);
    Vector2 thetaPhi = Utility::CartesianToSphericalUnit(dirZUp);

    // Normalize to generate UV [0, 1]
    // theta range [-pi, pi]
    float u = (thetaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
    // If we are at edge point (u == 1) make it zero since
    // piecewise constant function will not have that pdf (out of bounds)
    u = (u == 1.0f) ? 0.0f : u;
    // phi range [0, pi]
    float v = 1.0f - (thetaPhi[1] / MathConstants::Pi);
    // If (v == 1) then again pdf of would be out of bounds.
    // make it inbound
    v = (v == 1.0f) ? (v - MathConstants::SmallEpsilon) : v;
    localCoords = Vector2f(u, v);
}

__device__
inline void GPUCameraSpherical::GenerateRay(// Output
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
    // Create random location over sample pixel
    Vector2 randomOffset = (antiAliasOn)
                                ? Vector2(rng.Uniform(), rng.Uniform())
                                : Vector2(0.5f);

    // Normalize Coordinates X & Y = [0, 1]
    Vector2f normCoords = Vector2(static_cast<float>(sampleId[0]),
                                  static_cast<float>(sampleId[1]));
    normCoords += randomOffset;
    normCoords /= Vector2(static_cast<float>(sampleMax[0]),
                          static_cast<float>(sampleMax[1]));

    // Calculate Spherical Coordinates
    Vector2f sphericalCoords = Vector2f(// [-pi, pi]
                                        (normCoords[0] * MathConstants::Pi * 2.0f) - MathConstants::Pi,
                                        // [0, pi]
                                        (1.0f - normCoords[1]) * MathConstants::Pi);

    // Incorporate pixel ratio
    // TODO: Incorporate Pix Ratio (for non-square textures)
    sphericalCoords[0] *= pixelRatio;

    // Normalized spherical coords are local space coords
    localCoords = normCoords;

    // Convert to Cartesian
    Vector3 dirZUp = Utility::SphericalToCartesianUnit(sphericalCoords);
    Vector3 dirYUp = Vector3(dirZUp[1], dirZUp[2], dirZUp[0]);

    // Finally transform from local to world
    Matrix3x3 viewMat;
    TransformGen::Space(viewMat, right, up, direction);
    dirYUp = viewMat * dirYUp;

    // Initialize Ray
    ray.ray = RayF(dirYUp, position);
    ray.tMin = nearFar[0];
    ray.tMax = nearFar[1];
}

__device__
inline float GPUCameraSpherical::Pdf(const Vector3&,
                                     const Vector3&) const
{
    return 0.0f;
}

__device__
inline float GPUCameraSpherical::Pdf(float,
                                     const Vector3&,
                                     const Vector3&,
                                     const QuatF&) const
{
    return 0.0f;
}

__device__
inline uint32_t GPUCameraSpherical::FindPixelId(const RayReg&,
                                                const Vector2i&) const
{
    // TODO:
    return 0;
}

inline __device__ bool GPUCameraSpherical::CanBeSampled() const
{
    return false;
}

__device__
inline Matrix4x4 GPUCameraSpherical::VPMatrix() const
{
    // Well we cant generate Projection Matrix (or a Matrix)
    // for cartesian => spherical only return view matrix
    Matrix3x3 viewMatrix;
    TransformGen::Space(viewMatrix, right, up, direction);
    return ToMatrix4x4(viewMatrix);
}

__device__
inline Vector2f GPUCameraSpherical::NearFar() const
{
    return nearFar;
}

__device__
inline VisorTransform GPUCameraSpherical::GenVisorTransform() const
{
    return VisorTransform
    {
        position,
        position + direction,
        up
    };
}

__device__
inline void GPUCameraSpherical::SwapTransform(const VisorTransform& t)
{
    // Assume these are already transformed
    position = t.position;
    direction = (t.gazePoint - t.position).Normalize();
    up = t.up;

    // Camera Vector Correction
    right = Cross(direction, up).Normalize();
    up = Cross(right, direction).Normalize();
    direction = Cross(up, right).Normalize();
}

__device__
inline GPUCameraPixel GPUCameraSpherical::GeneratePixelCamera(const Vector2i& pixelId,
                                                              const Vector2i& resolution) const
{
    const auto ZERO3 = Zero3;
    const auto ZERO2 = Zero2;
    // TODO: Implement
    return GPUCameraPixel(ZERO3,
                          ZERO3,
                          ZERO3,
                          ZERO3,
                          ZERO2,
                          ZERO3,
                          pixelId,
                          resolution,
                          mediumIndex,
                          workKey,
                          gTransform);
}

inline CPUCameraGroupSpherical::CPUCameraGroupSpherical(const GPUPrimitiveGroupI* pg)
    : Base(*pg)
{}

inline const char* CPUCameraGroupSpherical::Type() const
{
    return TypeName();
}


inline size_t CPUCameraGroupSpherical::UsedCPUMemory() const
{
    size_t totalSize = (Base::UsedCPUMemory() +
                        sizeof(Data) * hCameraData.size());
    return totalSize;
}

static_assert(IsTracerClass<CPUCameraGroupSpherical>::value,
              "CPUCameraSpherical is not a tracer class");