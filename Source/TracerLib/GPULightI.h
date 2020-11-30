#pragma once

#include "GPUEndpointI.h"

#include "RayLib/Constants.h"
#include "RayLib/SceneStructs.h"
#include "CudaConstants.h"

#include "NodeListing.h"

#include <type_traits>

//#include "RayLib/HybridFunctions.h"
//#include "RayLib/Quaternion.h"

class GPUTransformI;
class RandomGPU;

using GPULightI = GPUEndpointI;
using GPULightList = std::vector<const GPULightI*>;

class CPULightGroupI
{
	protected:
		// Global Transform Array
		const GPUTransformI** dGPUTransforms;
	public:
										CPULightGroupI();
		virtual							~CPULightGroupI() = default;

		// Interface
		virtual const char*				Type() const = 0;
		virtual const GPULightList&		GPULights() const = 0;
		virtual SceneError				InitializeGroup(const ConstructionDataList& lightNodes,
                                                        const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                        const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                        const MaterialKeyListing& allMaterialKeys,
														double time,
														const std::string& scenePath) = 0;
		virtual SceneError				ChangeTime(const NodeListing& lightNodes, double time,
												   const std::string& scenePath) = 0;
		virtual TracerError				ConstructLights(const CudaSystem&) = 0;
		virtual uint32_t				LightCount() const = 0;

		virtual size_t					UsedGPUMemory() const = 0;
		virtual size_t					UsedCPUMemory() const = 0;	

		void							AttachGlobalTransformArray(const GPUTransformI**);
};

inline CPULightGroupI::CPULightGroupI()
	: dGPUTransforms(nullptr)
{}

inline void CPULightGroupI::AttachGlobalTransformArray(const GPUTransformI** deviceTranfsorms)
{
	dGPUTransforms = deviceTranfsorms;
}


//class DiskLight final : public GPULightI
//{
//    private:
//        // Sample Ready Parameters
//        // All of which is world space
//        Vector3             center;
//        Vector3             normal;
//        float               radius;
//        float               area;
//
//    protected:
//    public:
//        // Constructors & Destructor
//        __device__          DiskLight(const Vector3& center,
//                                      const Vector3& normal,
//                                      float radius,
//                                      const GPUDistribution2D* lumDist,
//                                      HitKey k,
//                                      PrimitiveId pId,
//                                      uint16_t mediumIndex);
//
//        // Interface 
//        __device__ void     Sample(// Output
//                                   float& distance,
//                                   Vector3& direction,
//                                   float& pdf,
//                                   // Input
//                                   const Vector3& position,
//                                   // I-O
//                                   RandomGPU&) const override;
//
//        __device__ void     GenerateRay(// Output
//                                        RayReg&,
//                                        // Input
//                                        const Vector2i& sampleId,
//                                        const Vector2i& sampleMax,
//                                        // I-O
//                                        RandomGPU&) const override;
//};
//
//class SphericalLight final : public GPULightI
//{
//    private:
//        // Sample Ready Parameters
//        // All of which is world space
//        Vector3             center;
//        float               radius;
//        float               area;
//
//    protected:
//    public:
//        // Constructors & Destructor
//        __device__          SphericalLight(const Vector3& center,
//                                           float radius,
//                                           const GPUDistribution2D* lumDist,
//                                           HitKey k,
//                                           PrimitiveId pId,
//                                           uint16_t mediumIndex);
//
//        // Interface 
//        __device__ void     Sample(// Output
//                                   float& distance,
//                                   Vector3& direction,
//                                   float& pdf,
//                                   // Input
//                                   const Vector3& position,
//                                   // I-O
//                                   RandomGPU&) const override;
//
//        __device__ void     GenerateRay(// Output
//                                        RayReg&,
//                                        // Input
//                                        const Vector2i& sampleId,
//                                        const Vector2i& sampleMax,
//                                        // I-O
//                                        RandomGPU&) const override;
//};


//
//__device__
//inline DiskLight::DiskLight(const Vector3& center,
//                            const Vector3& normal,
//                            float radius,
//                            const GPUDistribution2D* lumDist,
//                            HitKey k,
//                            PrimitiveId pId,
//                            uint16_t mediumIndex)
//    : GPULightI(lumDist, k, pId, mediumIndex)
//    , center(center)
//    , normal(normal)
//    , radius(radius)
//    , area(MathConstants::Pi * radius * radius)
//{}
//
//__device__
//inline SphericalLight::SphericalLight(const Vector3& center,
//                                      float radius,
//                                      const GPUDistribution2D* lumDist,
//                                      HitKey k,
//                                      PrimitiveId pId,
//                                      uint16_t materialIndex)
//    : GPULightI(lumDist, k, pId, materialIndex)
//    , center(center)
//    , radius(radius)
//    , area (MathConstants::Pi * radius * radius * 4.0f)
//{}

//__device__
//inline void DiskLight::Sample(// Output
//                              float& distance,
//                              Vector3& direction,
//                              float& pdf,
//                              // Input
//                              const Vector3& worldLoc,
//                              // I-O
//                              RandomGPU& rng) const
//{
//    float r = GPUDistribution::Uniform<float>(rng) * radius;
//    float tetha = GPUDistribution::Uniform<float>(rng) * 2.0f * MathConstants::Pi;
//
//    // Aligned to Axis Z
//    Vector3 disk = Vector3(sqrt(r) * cos(tetha),
//                               sqrt(r) * sin(tetha),
//                               0.0f);               
//    // Rotate to disk normal
//    QuatF rotation = Quat::RotationBetweenZAxis(normal);
//    Vector3 worldDisk = rotation.ApplyRotation(disk);
//    Vector3 position = center + worldDisk;
//
//    direction = position - worldLoc;
//    float distanceSqr = direction.LengthSqr();
//    distance = sqrt(distanceSqr);
//    direction *= (1.0f / distance);
//
//    float nDotL = max(normal.Dot(-direction), 0.0f);
//    pdf = distanceSqr / (nDotL * area);
//}
//

//// ========================================= 
//__device__
//inline void SphericalLight::Sample(// Output
//                                   float& distance,
//                                   Vector3& direction,
//                                   float& pdf,
//                                   // Input
//                                   const Vector3& worldLoc,
//                                   // I-O
//                                   RandomGPU& rng) const
//{
//    // Marsaglia 1972
//    // http://mathworld.wolfram.com/SpherePointPicking.html
//
//    float x1 = GPUDistribution::Uniform<float>(rng) * 2.0f - 1.0f;
//    float x2 = GPUDistribution::Uniform<float>(rng) * 2.0f - 1.0f;
//
//    float x1Sqr = x1 * x1;
//    float x2Sqr = x2 * x2;
//    float coeff = sqrt(1 - x1Sqr - x2Sqr);
//
//    Vector3 unitSphr = Vector3(2.0f * x1 * coeff,
//                               2.0f * x2 * coeff,
//                               1.0f - 2.0f * (x1Sqr + x2Sqr));
//
//    Vector3 position = center + radius * unitSphr;
//    direction = position - worldLoc;
//    distance = direction.LengthSqr();
//    pdf = distance / area;
//    distance = sqrt(distance);
//    direction *= (1.0f / distance);
//}