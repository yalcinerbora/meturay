#pragma once

#include "TracerLib/TracerLogicP.cuh"

struct RayAuxBasic
{
	Vector3f		totalRadiance;
	uint32_t		pixelId;
	uint32_t		pixelSampleId;
};

__device__ __host__
inline void RayInitBasic(RayAuxBasic* gOutBasic,
						 const uint32_t writeLoc,
						 // Input
						 const RayAuxBasic& defaults,
						 const RayReg& ray,
						 // Index
						 const uint32_t localPixelId,
						 const uint32_t pixelSampleId)
{
	RayAuxBasic init = defaults;
	init.pixelId = localPixelId;
	init.pixelSampleId = pixelSampleId;

	gOutBasic[writeLoc] = init;
}

//__device__ __host__
//inline void RayFinalizeBasic(// Output
//							 Vector4* gImage,
//							 // Input
//							 const RayAuxBasic& auxData,
//							 const RayReg& ray,
//							 //
//							 RandomGPU& rng)
//{
//	gImage[auxData.pixelId][0] = auxData.totalRadiance[0];
//	gImage[auxData.pixelId][1] = auxData.totalRadiance[1];
//	gImage[auxData.pixelId][2] = auxData.totalRadiance[2];
//}

class TracerBasic : public TracerBaseLogic<RayAuxBasic, RayInitBasic>
{
	private:
		static constexpr RayAuxBasic	initals = {Zero3f};

	protected:
	public:
		// Constructors & Destructor
						TracerBasic(GPUBaseAcceleratorI& ba,
									AcceleratorGroupList&& ag,
									AcceleratorBatchMappings&& ab,
									MaterialGroupList&& mg,
									MaterialBatchMappings&& mb,
									//
									const TracerParameters& parameters,
									uint32_t hitStructSize,
									const Vector2i maxMats,
									const Vector2i maxAccels);
						~TracerBasic() = default;

		TracerError		Initialize() override;

		size_t			GenerateRays(RayMemory&, RNGMemory&,
									 const GPUScene& scene,
									 int cameraId,
									 int samplePerLocation,
									 Vector2i resolution,
									 Vector2i pixelStart = Zero2i,
									 Vector2i pixelEnd = BaseConstants::IMAGE_MAX_SIZE) override;
};