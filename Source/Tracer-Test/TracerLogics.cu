#include "TracerLogics.cuh"
#include "RayLib/TracerError.h"

TracerBasic::TracerBasic(const GPUBaseAcceleratorI& baseAccelerator,
						 const AcceleratorBatchMappings& a,
						 const MaterialBatchMappings& m,
						 const TracerOptions& options)
	:TracerBaseLogic(baseAccelerator, a, m, options, initals)
{}

TracerError TracerBasic::Initialize()
{
	return TracerError::OK;
}

void TracerBasic::GenerateRays(RayMemory&, RNGMemory&,
							   const uint32_t rayCount)
{

}