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
class GPUTracerI;

using NameGPUPair = std::pair<std::string, const CudaGPU*>;

using GPUTracerPtr = SharedLibPtr<GPUTracerI>;
using GPUBaseAccelPtr = SharedLibPtr<GPUBaseAcceleratorI>;
using GPUAccelGPtr = SharedLibPtr<GPUAcceleratorGroupI>;
using GPUPrimGPtr = SharedLibPtr<GPUPrimitiveGroupI>;
using GPUMatGPtr = SharedLibPtr<GPUMaterialGroupI>;

using MatPrimPair = std::pair<const GPUPrimitiveGroupI*, const GPUMaterialGroupI*>;

// Kernel Mappings
using AcceleratorBatchMap = std::map<uint32_t, GPUAcceleratorGroupI*>;
using WorkBatchMap = std::map<uint32_t, GPUWorkBatchI*>;
using WorkBatchCreationInfo = std::map<uint32_t, MatPrimPair>;

// Constant Paramters that cannot be changed after initialization
struct TracerParameters
{
    bool verbose;
    uint32_t seed;
};