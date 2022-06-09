#pragma once

#include "RayLib/Vector.h"
#include "RayLib/AABB.h"
#include "RayLib/HitStructs.h"
#include "RayLib/SceneStructs.h"

#include "NodeListing.h"

struct SceneError;
struct TracerError;

class GPUBoundaryMaterialGroupI;
class GPUEndpointI;
class GPUTransformI;
class CudaSystem;
class RNGeneratorGPUI;

struct RayReg;

using GPUEndpointList = std::vector<const GPUEndpointI*>;
using KeyMaterialMap = std::map<uint32_t, const GPUBoundaryMaterialGroupI*>;

// Endpoint interface
// it is either camera or a light source
// Only dynamic polymorphism case for tracer
class GPUEndpointI
{
    protected:
        // Medium of the endpoint
        // used to initialize rays when generated
        uint16_t                mediumIndex;
        // Transform of this endpoint
        const GPUTransformI&    gTransform;
        // Unique Endpoint Id
        uint32_t                endpointId;
        // Work key
        HitKey                  workKey;

    public:
        __device__              GPUEndpointI(uint16_t mediumIndex,
                                             HitKey, const GPUTransformI&);
        virtual                 ~GPUEndpointI() = default;

        // Interface
        // Sample the endpoint from a point
        // Return directional pdf
        // Direction is NOT normalized (and should not be normalized)
        // that data may be useful (when sampling a light source using NEE)
        // tMax (distance) can be generated from it
        virtual __device__ void     Sample(// Output
                                           float& distance,
                                           Vector3& direction,
                                           float& pdf,
                                           // Input
                                           const Vector3& position,
                                           // I-O
                                           RNGeneratorGPUI&) const = 0;
        // Generate a Ray from this endpoint
        virtual __device__ void     GenerateRay(// Output
                                                RayReg&,
                                                // Input
                                                const Vector2i& sampleId,
                                                const Vector2i& sampleMax,
                                                // I-O
                                                RNGeneratorGPUI&,
                                                // Options
                                                bool antiAliasOn = true) const = 0;
        virtual __device__ float    Pdf(const Vector3& direction,
                                        const Vector3& position) const = 0;
        virtual __device__ float    Pdf(float distance,
                                        const Vector3& hitPosition,
                                        const Vector3& direction,
                                        const QuatF& tbnRotation) const = 0;
        virtual __device__ bool     CanBeSampled() const = 0;

        // Checkers
        __device__ bool             operator==(uint32_t endpointId) const;
        __device__ bool             operator!=(uint32_t endpointId) const;

        // Getters & Setters
        __device__ void                     SetEndpointId(uint32_t);

        __device__ uint32_t                 EndpointId() const;
        __device__ uint16_t                 MediumIndex() const;
        __device__ const GPUTransformI&     Transform() const;
        __device__ HitKey                   WorkKey() const;
};

class CPUEndpointGroupI
{
	public:
		virtual								~CPUEndpointGroupI() = default;

		// Interface
		virtual const char*					Type() const = 0;
		virtual const GPUEndpointList&		GPUEndpoints() const = 0;
		virtual SceneError					InitializeGroup(const EndpointGroupDataList& endpointNodes,
                                                            const TextureNodeMap& textures,
											                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
											                const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
											                uint32_t batchId, double time,
															const std::string& scenePath) = 0;
		virtual SceneError					ChangeTime(const NodeListing& lightNodes, double time,
													   const std::string& scenePath) = 0;
		virtual TracerError					ConstructEndpoints(const GPUTransformI**,
                                                               const AABB3f& sceneAABB,
                                                               const CudaSystem&) = 0;
		virtual uint32_t					EndpointCount() const = 0;

        // This returns the packed keys,
        // if a primitive endpoint (light) present
        // it does only gives the first hit key of the primitive batch
        virtual const std::vector<HitKey>&  PackedHitKeys() const = 0;
        // Returns maximum used inner id number
        // It will be used to determine how many bits
        // should be used on radix sort
        virtual uint32_t                    MaxInnerId() const = 0;

		virtual size_t						UsedGPUMemory() const = 0;
		virtual size_t						UsedCPUMemory() const = 0;
};

__device__
inline GPUEndpointI::GPUEndpointI(uint16_t mediumIndex,
                                  HitKey hk,
                                  const GPUTransformI& gTransform)
    : mediumIndex(mediumIndex)
    , gTransform(gTransform)
    , endpointId(UINT32_MAX)
    , workKey(hk)
{}

__device__
inline bool GPUEndpointI::operator==(uint32_t eId) const
{
    return (endpointId == eId);
}

__device__
inline bool GPUEndpointI::operator!=(uint32_t eId) const
{
    return !(*this == eId);
}

__device__
inline void GPUEndpointI::SetEndpointId(uint32_t e)
{
    endpointId = e;
}

__device__
inline uint32_t GPUEndpointI::EndpointId() const
{
    return endpointId;
}

__device__
inline uint16_t GPUEndpointI::MediumIndex() const
{
    return mediumIndex;
}

__device__
inline const GPUTransformI& GPUEndpointI::Transform() const
{
    return gTransform;
}

__device__
inline HitKey GPUEndpointI::WorkKey() const
{
    return workKey;
}
