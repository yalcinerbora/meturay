#pragma once

#include "GPULightP.cuh"
#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"
#include "MangledNames.h"

class GPULightSpot final  : public GPULightP
{
    private:
        Vector3             position;
        float               cosMin;
        Vector3             direction;
        float               cosMax;

    protected:
    public:
        // Constructors & Destructor
        __device__              GPULightSpot(// Per Light Data
                                             const Vector3& position,
                                             const Vector3& direction,
                                             const Vector2& aperture,
                                             // Base Class Related
                                             const TextureRefI<2, Vector3f>& gRad,
                                             uint16_t mediumId, HitKey,
                                             const GPUTransformI& gTrans);
                                ~GPULightSpot() = default;
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

class CPULightGroupSpot final : public CPULightGroupP<GPULightSpot>
{
    public:
        TYPENAME_DEF(LightGroup, "Spot");

        static constexpr const char*    POSITION_NAME = "position";
        static constexpr const char*    DIRECTION_NAME = "direction";
        static constexpr const char*    CONE_APERTURE_NAME = "aperture";

        using Base = CPULightGroupP<GPULightSpot>;

    private:
        std::vector<Vector3f>           hPositions;
        std::vector<Vector3f>           hDirections;
        std::vector<Vector2f>           hCosines;

    protected:
    public:
        // Constructors & Destructor
                                    CPULightGroupSpot(const GPUPrimitiveGroupI*,
                                                      const CudaGPU&);
                                    ~CPULightGroupSpot() = default;

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
inline GPULightSpot::GPULightSpot(// Per Light Data
                                  const Vector3& position,
                                  const Vector3& direction,
                                  const Vector2& aperture,
                                  // Base Class Related
                                  const TextureRefI<2, Vector3f>& gRad,
                                  uint16_t mediumId, HitKey hk,
                                  const GPUTransformI& gTrans)
    : GPULightP(gRad, mediumIndex, hk, gTrans)
    , position(gTrans.LocalToWorld(position))
    , cosMin(aperture[0])
    , direction(gTrans.LocalToWorld(direction, true).Normalize())
    , cosMax(aperture[1])
{}

__device__
inline void GPULightSpot::Sample(// Output
                                 float& distance,
                                 Vector3& dir,
                                 float& pdf,
                                 Vector2f& localCoords,
                                 // Input
                                 const Vector3& worldLoc,
                                 // I-O
                                 RNGeneratorGPUI&) const
{
    dir = -direction;
    distance = (position - worldLoc).Length();

    // Fake pdf to incorporate square falloff
    pdf = (distance * distance);

    // TODO: do localSpaceCoords
    localCoords = Vector2f(NAN, NAN);
}

__device__
inline void GPULightSpot::GenerateRay(// Output
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
inline float GPULightSpot::Pdf(const Vector3&,
                               const Vector3&) const
{
    return 0.0f;
}

__device__
inline float GPULightSpot::Pdf(float,
                               const Vector3&,
                               const Vector3&,
                               const QuatF&) const
{
    return 0.0f;
}

__device__
inline Vector3f GPULightSpot::GeneratePhoton(// Output
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
inline bool GPULightSpot::CanBeSampled() const
{
    return false;
}

inline CPULightGroupSpot::CPULightGroupSpot(const GPUPrimitiveGroupI* pg,
                                            const CudaGPU& gpu)
    : Base(*pg, gpu)
{}

inline const char* CPULightGroupSpot::Type() const
{
    return TypeName();
}

inline size_t CPULightGroupSpot::UsedCPUMemory() const
{
    size_t totalSize = (Base::UsedCPUMemory() +
                        hPositions.size() * sizeof(Vector3f) +
                        hDirections.size() * sizeof(Vector3f) +
                        hCosines.size() * sizeof(Vector2f));

    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupSpot>::value,
              "CPULightGroupSpot is not a tracer class");