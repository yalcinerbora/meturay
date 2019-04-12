#pragma once
/**

Structures that is related to TracerI

*/

#include <cstdint>
#include <vector>
#include <map>

class GPUAcceleratorBatchI;
class GPUMaterialBatchI;

using AcceleratorBatchMappings = std::map<uint32_t, GPUAcceleratorBatchI*>;
using MaterialBatchMappings = std::map<uint32_t, GPUMaterialBatchI*>;

struct MaterialOptions
{
	bool fullLoadTextures;
};

struct ShadeOpts
{
	int i;
};

struct HitOpts
{
	int j;
};

// Constant Paramters that cannot be changed after initialization time
struct TracerParameters
{
	uint32_t seed;
};

// Options that can be changed during runtime
struct TracerOptions
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