#pragma once

#include <vector>

#include "RayLib/MaterialI.h"
#include "RayLib/DeviceMemory.h"
#include "VolumeGPU.cuh"

struct RandomStackGMem;
struct HitRecord;

//
//
//class BasicMaterialGPU :public MaterialI, public BasicMaterialDeviceData
//{
//	private:
//		DeviceMemory					mem;
//
//		uint32_t						materialId;
//
//		Vector3f						diffuseAlbedo;
//		Vector3f						specularAlbedo;
//
//		Vector3f						
//
//	// Color Interpolation
//	const std::vector<Vector3f>		color;
//	const std::vector<float>		colorInterp;
//	// Opacity Interpolation
//	const std::vector<float>		opacity;
//	const std::vector<float>		opacityInterp;
//};


//
//
//struct FluidMaterialDeviceData
//{
//	struct RequiredMeshData
//	{
//		Vector3 position;
//		Vector3 normal;
//		Vector3 velocity;
//		float density;
//	};
//
//	// Color Interpolation
//	const Vector3f*				gColor;
//	const float*				gColorInterp;
//	// Opacity Interpolation
//	const float*				gOpacity;
//	const float*				gOpacityInterp;
//	// Transparency
//	const Vector3f				transparency;
//	// Volumetric Parameters
//	const float					absorbtionCoeff;
//	const float					scatteringCoeff;
//	// IoR
//	float						ior;
//
//	//// Bounces rays from records
//	//template <class T>
//	//__device__ void				Bounce(RayRecord outRays[],
//	//								   const HitRecord& inputHit,
//	//								   const RayRecord& inputRay,
//	//								   const RequiredMeshData& dataIn);
//
//	// Loads required material from the Mesh
//	template <class T, class Mesh>
//	__device__ void				FragmentData(T&, const Mesh&);
//
//};
//
//class FluidMaterialGPU : public MaterialI, public FluidMaterialDeviceData
//{
//	private:
//		DeviceMemory					mem;
//
//		uint32_t						materialId;
//
//		// Color Interpolation
//		const std::vector<Vector3f>		color;
//		const std::vector<float>		colorInterp;
//		// Opacity Interpolation
//		const std::vector<float>		opacity;
//		const std::vector<float>		opacityInterp;
//		
//	protected:
//	public:
//		// Constructors & Destructor
//										FluidMaterialGPU(uint32_t materialId,
//														 float indexOfRefraction,
//														 // Color Interpolation
//														 const std::vector<Vector3f>& color,
//														 const std::vector<float>& colorInterp,
//														 // Opacity Interpolation
//														 const std::vector<float>& opacity,
//														 const std::vector<float>& opacityInterp,
//														 // Transparency
//														 Vector3f transparency,
//														 // Volumetric Parameters
//														 float absorbtionCoeff,
//														 float scatteringCoeff);
//										FluidMaterialGPU(const FluidMaterialGPU&) = default;
//										FluidMaterialGPU(FluidMaterialGPU&&) = delete;
//		FluidMaterialGPU&				operator=(const FluidMaterialGPU&) = delete;
//		FluidMaterialGPU&				operator=(FluidMaterialGPU&&) = default;
//		
//		uint32_t						Id() const override;
//
//		void							BounceRays(// Outgoing Rays
//												   RayRecordGMem gOutRays,
//												   // Incoming Rays
//												   const ConstHitRecordGMem gHits,
//												   const ConstRayRecordGMem gRays,
//												   // Limits
//												   uint64_t rayCount,
//												   // Surfaces
//												   const Vector2ui* gSurfaceIndexList,
//												   const void* gSurfaces,
//												   SurfaceType) override;
//		Error							Load() override;
//		void							Unload() override;
//
//
//		// Data Fetch Functions
//		static __device__ void			FetchVolume(Vector3& velocity,
//													Vector3& normal,
//													Vector3& position,
//													float& density,
//													const HitRecord&,
//													const VolumeDeviceData&);
//};

//template <class DataFetchFunc, class T>
//__device__
//inline void FluidMaterialDeviceData::Bounce(RayRecord& reflectRecord,
//									 RayRecord& refractRecord,
//									 const HitRecord& hRecord,
//									 const RayRecord& rRecord,
//									 const T& surface)
//{
//
//	// Data Fetch
//	Vector3 velocity, normal, position;
//	float density;
//	DataFetchFunc(velocity, normal, position, density,
//				  hRecord, surface);
//
//	// Beer Term
//	Vector3 beerFactor = Vector3(absorbtionCoeff * density);
//	float distance = (position - rRecord.ray.getPosition()).Length();
//	beerFactor[0] = expf(logf(beerFactor[0]) * distance);
//	beerFactor[1] = expf(logf(beerFactor[1]) * distance);
//	beerFactor[2] = expf(logf(beerFactor[2]) * distance);
//
//	// All required data is available now do shade
//	// Volume normal sample can be zero (still on medium)
//	if(normal.Abs() >= Vector3(MathConstants::Epsilon))
//	{
//		normal.NormalizeSelf();
//
//		// Split
//		bool exitCase = rRecord.ray.getDirection().Dot(normal) > 0.0f;
//		Vector3 adjustedNormal = (exitCase) ? -normal : normal;
//		float hitMedium = (exitCase) ? 1.0f : ior;
//
//		// Total Reflection Case
//		Vector3 reflectionFactor = Vector3(1.0f);
//
//		RayF refractedRay;
//		bool refracted = rRecord.ray.Refract(refractedRay, adjustedNormal,
//											 rRecord.medium, hitMedium);
//		if(refracted)
//		{
//			// Fresnel Term
//			// Schclick's Approx
//			float cosTetha = (exitCase)
//								? -normal.Dot(refractedRay.getDirection())
//								: -normal.Dot(rRecord.ray.getDirection());
//			float r0 = (rRecord.medium - hitMedium) / (rRecord.medium + hitMedium);
//			float cosTerm = 1.0f - cosTetha;
//			cosTerm = cosTerm * cosTerm * cosTerm * cosTerm * cosTerm;
//			r0 = r0 * r0;
//			float fresnel = r0 + (1.0f - r0) * cosTerm;
//
//			beerFactor = (exitCase) ? Vector3(1.0f) : beerFactor;
//
//			// Energy Factors
//			Vector3 refractionFactor = Vector3(1.0f - fresnel) * beerFactor;
//			reflectionFactor = Vector3(fresnel) * beerFactor;
//
//			// Write this
//			refractedRay.NormalizeDirSelf();
//			refractedRay.AdvanceSelf(hRecord.distance + MathConstants::Epsilon);
//
//			// Record Save
//			refractRecord.ray = refractedRay;
//			refractRecord.medium = hitMedium;
//			refractRecord.pixelId = rRecord.pixelId;
//			refractRecord.sampleId = rRecord.sampleId;
//			refractRecord.totalRadiance = rRecord.totalRadiance * refractionFactor;
//		}
//
//		// Reflection
//		RayF reflectedRay = rRecord.ray.Reflect(adjustedNormal);
//
//		// Write this
//		reflectedRay.NormalizeDirSelf();
//		reflectedRay.AdvanceSelf(hRecord.distance + MathConstants::Epsilon);
//
//		refractRecord.ray = reflectedRay;
//		refractRecord.medium = rRecord.medium;
//		refractRecord.pixelId = rRecord.pixelId;
//		refractRecord.sampleId = rRecord.sampleId;
//		refractRecord.totalRadiance = rRecord.totalRadiance * reflectionFactor;
//	}
//	else
//	{
//		// Ray is traversing through same medium		
//		refractRecord.ray = rRecord.ray.Advance(hRecord.distance + MathConstants::Epsilon);
//		refractRecord.medium = rRecord.medium;
//		refractRecord.pixelId = rRecord.pixelId;
//		refractRecord.sampleId = rRecord.sampleId;
//		refractRecord.totalRadiance = rRecord.totalRadiance * beerFactor;
//	}
//}

//__device__
//inline void FluidMaterialGPU::FetchVolume(Vector3& velocity,
//										  Vector3& normal,
//										  Vector3& position,
//										  float& density,
//										  const HitRecord& record,
//										  const VolumeDeviceData& volume)
//{
//	//// Trilinear Interp
//	//Vector3f integral, fraction;
//	//fraction[0] = modff(record.baryCoord[0], &(integral[0]));
//	//fraction[1] = modff(record.baryCoord[1], &(integral[1]));
//	//fraction[2] = modff(record.baryCoord[2], &(integral[2]));
//	//Vector3ui index = static_cast<Vector3ui>(integral);
//
//	//Vector4f data[8];
//	//normal = Zero3;
//
//	//// Position Generation
//	////TODO:
//
//
//	//UNROLL_LOOP
//	//for(int i = 0; i < 8; i++)
//	//{
//	//	Vector3ui offset((i >> 0) & 0x1,
//	//					 (i >> 1) & 0x1,
//	//					 (i >> 2) & 0x1);
//	//	Vector3ui neighbourIndex = index + offset;
//
//	//	data[i] = volume.data[volume.LinearIndex(neighbourIndex)];
//
//	//	float density = data[i][3];
//	//	normal += (static_cast<Vector3f>(offset) * Vector3f(2.0f) - Vector3f(1.0f)) * density;
//	//}
//
//	//// Interpolate velocity & Density
//	//data[0] = Vector4f::Lerp(data[0], data[1], fraction[0]);
//	//data[1] = Vector4f::Lerp(data[2], data[3], fraction[0]);
//	//data[2] = Vector4f::Lerp(data[4], data[5], fraction[0]);
//	//data[3] = Vector4f::Lerp(data[6], data[7], fraction[0]);
//
//	//data[0] = Vector4f::Lerp(data[0], data[1], fraction[1]);
//	//data[1] = Vector4f::Lerp(data[2], data[3], fraction[1]);
//
//	//data[0] = Vector4f::Lerp(data[0], data[1], fraction[2]);
//
//
//	//// Out
//	//velocity = Vector3(data[0][0], data[0][1], data[0][2]);
//	//density = data[0][3];
//	//normal = -normal;
//}