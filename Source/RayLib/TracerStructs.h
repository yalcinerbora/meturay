#pragma once
/**

Structures that is related to TracerI

*/

#include <cstdint>
#include <vector>
#include <map>

#include "ObjectFuncDefinitions.h"

class CudaGPU;

class GPUBaseAcceleratorI;
class GPUAcceleratorGroupI;
class GPUPrimitiveGroupI;
class GPUMaterialGroupI;
class GPUWorkBatchI;

using NameGPUPair = std::pair<std::string, const CudaGPU*>;

using GPUBaseAccelPtr = SharedLibPtr<GPUBaseAcceleratorI>;
using GPUAccelGPtr = SharedLibPtr<GPUAcceleratorGroupI>;
using GPUPrimGPtr = SharedLibPtr<GPUPrimitiveGroupI>;
using GPUMatGPtr = SharedLibPtr<GPUMaterialGroupI>;

// Kernel Mappings
using AcceleratorBatchMappings = std::map<uint32_t, GPUAcceleratorGroupI*>;

// Constant Paramters that cannot be changed after initialization
struct TracerParameters
{
    uint32_t seed;
};

// Options that can be changed during runtime
struct TracerOptions
{
    // Misc
    bool        verbose;
};