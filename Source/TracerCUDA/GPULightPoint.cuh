#pragma once

#include "GPULightP.cuh"
#include "GPUTransformI.h"
#include "DeviceMemory.h"
#include "TypeTraits.h"

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
                                              uint16_t mediumId,
                                              const GPUTransformI& gTrans);
                                ~GPULightPoint() = default;
        // Interface
        __device__ void         Sample(// Output
                                       float& distance,
                                       Vector3& direction,
                                       float& pdf,
                                       // Input
                                       const Vector3& worldLoc,
                                       // I-O
                                       RandomGPU&) const override;

        __device__ void         GenerateRay(// Output
                                            RayReg&,
                                            // Input
                                            const Vector2i& sampleId,
                                            const Vector2i& sampleMax,
                                            // I-O
                                            RandomGPU&,
                                            // Options
                                            bool antiAliasOn = true) const override;

        __device__ float        Pdf(const Vector3& direction,
                                    const Vector3& position) const override;

        __device__ bool         CanBeSampled() const override;
};

class CPULightGroupPoint final : public CPULightGroupP<GPULightPoint>
{
    public:
        static constexpr const char*    TypeName(){return "Point"; }

        static constexpr const char*    NAME_POSITION = "position";

    private:
        std::vector<Vector3f>           hPositions;

    protected:
    public:
        // Cosntructors & Destructor
                                    CPULightGroupPoint(const CudaGPU& gpu,
                                                       const GPUPrimitiveGroupI*);
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
                                                       const CudaSystem&) override;
		
		size_t					    UsedCPUMemory() const override;
};

__device__
inline GPULightPoint::GPULightPoint(// Per Light Data
                                    const Vector3f& position,
                                    // Base Class Related
                                    const TextureRefI<2, Vector3f>& gRad,
                                    uint16_t mediumId,
                                    const GPUTransformI& gTrans)
    : GPULightP(gRad, mediumIndex, gTransform)
    , position(gTrans.LocalToWorld(position))
{}

__device__
inline void GPULightPoint::Sample(// Output
                                  float& distance,
                                  Vector3& direction,
                                  float& pdf,
                                  // Input
                                  const Vector3& worldLoc,
                                  // I-O
                                  RandomGPU&) const
{
    direction = (position - worldLoc);
    distance = direction.Length();
    direction *= (1.0f / distance);

    // Fake pdf to incorporate square faloff
    pdf = (distance * distance);
}

__device__
inline void GPULightPoint::GenerateRay(// Output
                                       RayReg&,
                                       // Input
                                       const Vector2i& sampleId,
                                       const Vector2i& sampleMax,
                                       // I-O
                                       RandomGPU&,
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
inline bool GPULightPoint::CanBeSampled() const
{
    return false;
}

inline CPULightGroupPoint::CPULightGroupPoint(const CudaGPU& gpu,
                                              const GPUPrimitiveGroupI*)
    : CPULightGroupP<GPULightPoint>(gpu)    
{}

inline const char* CPULightGroupPoint::Type() const
{
    return TypeName();
}

inline size_t CPULightGroupPoint::UsedCPUMemory() const
{
    size_t totalSize = (CPULightGroupP<GPULightPoint>::UsedCPUMemory() +
                        hPositions.size() * sizeof(Vector3f));
    return totalSize;
}

static_assert(IsTracerClass<CPULightGroupPoint>::value,
              "CPULightGroupPoint is not a tracer class");