#pragma once

#include "TracerLib/TracerLogicP.cuh"

struct RayAuxBasic
{
	Vector3f	totalRadiance;
	uint32_t	pixelId;
};

__device__ __host__
inline void RayInitBasic(RayAuxBasic* gOutBasic,
						 const uint32_t writeLoc,
						 // Input
						 const RayAuxBasic& defaults,
						 const RayReg& ray,
						 // Index
						 const Vector2ui& globalPixelId,
						 const Vector2ui& localSampleId,
						 const uint32_t samplePerPixel)
{
	gOutBasic[writeLoc] = defaults;
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
										TracerBasic(GPUBaseAcceleratorI& baseAccelerator,
													const AcceleratorBatchMappings&,
													const MaterialBatchMappings&,
													const TracerParameters& parameters,
													uint32_t hitStructSize,													
													const Vector2i maxMats,
													const Vector2i maxAccels);
										~TracerBasic() = default;

		TracerError						Initialize() override;

		void							GenerateRays(RayMemory&, RNGMemory&,
													 const uint32_t rayCount) override;
};