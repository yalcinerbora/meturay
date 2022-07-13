#pragma once

#include "GPULightP.cuh"
#include "GPUTransformI.h"
#include "TypeTraits.h"
#include "MangledNames.h"

class GPULightPoint final : public GPULightP
{
    private:
        Vector3f                position;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightPoint(// Per Light Data
                                              const Vector3f& position,
                                              // Base Class Related
                                              const TextureRefI<2, Vector3f>& gRad,
                                              uint16_t mediumId, HitKey,
                                              const GPUTransformI& gTrans);
                                ~GPULightPoint() = default;
        // Interface
        __device__ void         Sample(// Output
                                       float& distance,
                                       Vector3& direction,
                                       float& pdf,
                                       Vector2f& localCoords,
                                       // Input
                                       const Vector3& worldLoc,
                                       // I-O
                                       RNGeneratorGPUI&) const override;

        __device__ void         GenerateRay(// Output
                                            RayReg&,
                                            Vector2f& localCoords,
                                            // Input
                                            const Vector2i& sampleId,
                                            const Vector2i& sampleMax,
                                            // I-O
                                            RNGeneratorGPUI&,
                                            // Options
                                            bool antiAliasOn = true) const override;

        __device__ float        Pdf(const Vector3& direction,
                                    const Vector3& position) const override;

        __device__ float        Pdf(float distance,
                                    const Vector3& hitPosition,
                                    const Vector3& direction,
                                    const QuatF& tbnRotation) const override;

        // Photon Stuff
        __device__ Vector3f     GeneratePhoton(// Output
                                               RayReg& rayOut,
                                               Vector3f& normal,
                                               float& posPDF,
                                               float& dirPDF,
                                               // I-O
                                               RNGeneratorGPUI&) const override;

        __device__ bool         CanBeSampled() const override;
};

class CPULightGroupPoint final : public CPULightGroupP<GPULightPoint>
{
    public:
        TYPENAME_DEF(LightGroup, "Point");

        static constexpr const char*    POSITION_NAME = "position";

        using Base = CPULightGroupP<GPULightPoint>;

    private:
        std::vector<Vector3f>           hPositions;

    protected:
    public:
        // Constructors & Destructor
                                    CPULightGroupPoint(const GPUPrimitiveGroupI*,
                                                       const CudaGPU&);
                                    ~CPULightGroupPoint() = default;

        const char*				    Type() const override;
		SceneError				    InitializeGroup(const EndpointGroupDataList& endpointNodes,
                                                    const TextureNodeMap& textures,
                                                    const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                    const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                    uint32_t batchId, double time,
                                                    const std::string& scenePath) override;
        SceneError				    ChangeTime(const NodeListing& lightNodes, double time,
                                               const std::string& scenePath) override;
        TracerError				    ConstructEndpoints(const GPUTransformI**,
                                                       const AABB3f&,
                                                       const CudaSystem&) override;

		size_t					    UsedCPUMemory() const override;
};

__device__
inline GPULightPoint::GPULightPoint(// Per Light Data
                                    const Vector3f& position,
                                    // Base Class Related
                                    const TextureRefI<2, Vector3f>& gRad,
                                    uint16_t mediumId, HitKey hk,
                                    const GPUTransformI& gTrans)
    : GPULightP(gRad, mediumIndex, hk, gTransform)
    , position(gTrans.LocalToWorld(position))
{}

__device__
inline void GPULightPoint::Sample(// Output
                                  float& distance,
                                  Vector3& direction,
                                  float& pdf,
                                  Vector2f& localCoords,
                                  // Input
                                  const Vector3& worldLoc,
                                  // I-O
                                  RNGeneratorGPUI&) const
{
    direction = (position - worldLoc);
    distance = direction.Length();
    direction *= (1.0f / distance);

    // Since point light has no surface not local coords
    localCoords = Vector2f(NAN, NAN);

    // Fake pdf to incorporate square falloff
    pdf = (distance * distance);
}

__device__
inline void GPULightPoint::GenerateRay(// Output
                                       RayReg&,
                                       Vector2f&,
                                       // Input
                                       const Vector2i& sampleId,
                                       const Vector2i& sampleMax,
                                       // I-O
                                       RNGeneratorGPUI&,
                                       // Options
                                       bool antiAliasOn) const
{
    // TODO: Implement
}

__device__
inline float GPULightPoint::Pdf(const Vector3& worldDir,
                                const Vector3& worldPos) const
{
    return 0.0f;
}

__device__
inline float GPULightPoint::Pdf(float distance,
                                const Vector3& hitPosition,
                                const Vector3& direction,
                                const QuatF& tbnRotation) const
{
    return 0.0f;
}

__device__
inline Vector3f GPULightPoint::GeneratePhoton(// Output
                                              RayReg& rayOut,
                                              Vector3f& normal,
                                              float& posPDF,
                                              float& dirPDF,
                                              // I-O
                                              RNGeneratorGPUI& rng) const
{
    // TODO: Implement
    return Zero3f;
}

__device__
inline bool GPULightPoint::CanBeSampled() const
{
    return false;
}

inline CPULightGroupPoint::CPULightGroupPoint(const GPUPrimitiveGroupI* pg,
                                              const CudaGPU& gpu)
    : Base(*pg, gpu)
{}

inline const char* CPULightGroupPoint::Type() const
{
    return TypeName();
}

inline size_t CPULightGroupPoint::UsedCPUMemory() const
{
    size_t totalSize = (Base::UsedCPUMemory() +
                        hPositions.size() * sizeof(Vector3f));
    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupPoint>::value,
              "CPULightGroupPoint is not a tracer class");