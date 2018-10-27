#pragma once

#include <map>
#include <cstdint>

class GPUAcceleratorBatchI;
class GPUMaterialBatchI;

using AcceleratorBatchMappings = std::map<uint32_t, GPUAcceleratorBatchI*>;
using MaterialBatchMappings = std::map<uint32_t, GPUMaterialBatchI*>;

struct ShadeOpts
{
	int i;
};

struct HitOpts
{
	int j;
};

struct TracerOptions
{
//	Vector2i		materialKeyRange;
//	Vector2i		acceleratorKeyRange;

	uint32_t		seed;
	uint32_t		hitStructMaxSize;
};