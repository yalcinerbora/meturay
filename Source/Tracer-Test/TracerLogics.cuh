#pragma once

#include "TracerLib/TracerLogicP.cuh"

struct RayAuxBasic
{
	Vector3f totalIrradiance;
};

__device__ __host__
inline void RayInitBasic(RayAuxBasic* gOutBasic,
						 const uint32_t writeLoc,
						 // Input
						 const RayAuxBasic defaults,
						 // Index
						 const Vector2ui& globalPixelId,
						 const Vector2ui& localSampleId,
						 const uint32_t samplePerPixel)
{
	gOutBasic[writeLoc] = defaults;
}

class TracerBasic : public TracerBaseLogic<RayAuxBasic, RayInitBasic>
{
	private:
		static constexpr RayAuxBasic	initals = {Zero3f};

	protected:
	public:
		// Constructors & Destructor
										TracerBasic(const GPUBaseAcceleratorI& baseAccelerator,
													const AcceleratorBatchMappings&,
													const MaterialBatchMappings&,
													const TracerOptions& options);
										~TracerBasic() = default;

		TracerError						Initialize() override;

		void							GenerateRays(RayMemory&, RNGMemory&,
													 const uint32_t rayCount) override;
};