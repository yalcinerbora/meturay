#pragma once
/**

Structures that is related to TracerI

*/

#include <cstdint>
#include <vector>

struct TracerParameters
{
	uint32_t		depth;
	uint32_t		sampleCount;
};

struct MatBatchRayDataCPU
{
	uint32_t				batchId;
	std::vector<uint8_t>	record;
	//
	uint64_t				raysOffset;
	uint64_t				auxiliaryOffset;
	uint64_t				primitiveIdsOffset;
	uint64_t				hitStructOffset;	
};