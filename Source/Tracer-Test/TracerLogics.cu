#include "TracerLogics.cuh"
#include "RayLib/TracerError.h"

TracerBasic::TracerBasic(GPUBaseAcceleratorI& baseAccelerator,
						 const AcceleratorBatchMappings& a,
						 const MaterialBatchMappings& m,
						 const TracerParameters& options,
						 uint32_t hitStructSize,
						 const Vector2i maxMats,
						 const Vector2i maxAccels)
	: TracerBaseLogic(baseAccelerator, 
					  a, m, 
					  options, 
					  initals,
					  hitStructSize,
					  maxMats,
					  maxAccels)
{}

TracerError TracerBasic::Initialize()
{
	return TracerError::OK;
}

void TracerBasic::GenerateRays(RayMemory&, RNGMemory&,
							   const uint32_t rayCount)
{

}