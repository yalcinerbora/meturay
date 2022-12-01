#pragma once

#include "GPULightP.cuh"
#include "GPUTransformI.h"
#include "TypeTraits.h"
#include "MangledNames.h"

class GPULightConstant final : public GPULightP
{
    private:
    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightConstant(// Base Class Related
                                                 const TextureRefI<2, Vector3f>& gRad,
                                                 uint16_t mediumId, HitKey,
                                                 const GPUTransformI& gTrans);
                                ~GPULightConstant() = default;
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
                                            Vector2f&,
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

        __device__ Vector3f     GeneratePhoton(// Output
                                               RayReg& rayOut,
                                               Vector3f& normal,
                                               float& posPDF,
                                               float& dirPDF,
                                               // I-O
                                               RNGeneratorGPUI& rng) const override;

        __device__ bool         CanBeSampled() const override;
};

class CPULightGroupConstant final : public CPULightGroupP<GPULightConstant>
{
    public:
        TYPENAME_DEF(LightGroup, "Constant");

        using Base = CPULightGroupP<GPULightConstant>;

    private:
    protected:
    public:
        // Constructors & Destructor
                                    CPULightGroupConstant(const GPUPrimitiveGroupI*,
                                                       const CudaGPU&);
                                    ~CPULightGroupConstant() = default;

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
};

__device__
inline GPULightConstant::GPULightConstant(// Base Class Related
                                          const TextureRefI<2, Vector3f>& gRad,
                                          uint16_t mediumId, HitKey hk,
                                          const GPUTransformI& gTrans)
    : GPULightP(gRad, mediumIndex, hk, gTransform)
{}

__device__
inline void GPULightConstant::Sample(// Output
                                     float& distance,
                                     Vector3& direction,
                                     float& pdf,
                                     Vector2f& localCoords,
                                     // Input
                                     const Vector3& worldLoc,
                                     // I-O
                                     RNGeneratorGPUI&) const
{
    distance = FLT_MAX;
    direction = Zero3f;
    pdf = 0.0f;
    localCoords = Vector2f(NAN, NAN);
}

__device__
inline void GPULightConstant::GenerateRay(// Output
                                          RayReg&,
                                          Vector2f&,
                                          // Input
                                          const Vector2i& sampleId,
                                          const Vector2i& sampleMax,
                                          // I-O
                                          RNGeneratorGPUI&,
                                          // Options
                                          bool antiAliasOn) const
{}

__device__
inline float GPULightConstant::Pdf(const Vector3& worldDir,
                                   const Vector3& worldPos) const
{
    return 0.0f;
}

__device__
inline float GPULightConstant::Pdf(float distance,
                                   const Vector3& hitPosition,
                                   const Vector3& direction,
                                   const QuatF& tbnRotation) const
{
    return 0.0f;
}

__device__
inline Vector3f GPULightConstant::GeneratePhoton(// Output
                                                 RayReg& rayOut,
                                                 Vector3f& normal,
                                                 float& posPDF,
                                                 float& dirPDF,
                                                 // I-O
                                                 RNGeneratorGPUI& rng) const
{
    posPDF = 0.0f;
    dirPDF = 0.0f;
    normal = Zero3f;
    rayOut.tMin = 0.0f;
    rayOut.tMax = 0.0f;
    rayOut.ray = RayF(Vector3f(0.0f), Vector3f(0.0f));
    return Zero3f;
}

__device__
inline bool GPULightConstant::CanBeSampled() const
{
    return false;
}

inline CPULightGroupConstant::CPULightGroupConstant(const GPUPrimitiveGroupI* pg,
                                                    const CudaGPU& gpu)
    : Base(*pg, gpu)
{}

inline const char* CPULightGroupConstant::Type() const
{
    return TypeName();
}

static_assert(IsTracerClass<CPULightGroupConstant>::value,
              "CPULightGroupConstant is not a tracer class");