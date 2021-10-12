#pragma once
/**

Structures that is related to TracerI

*/

#include <cstdint>
#include <vector>
#include <map>
#include <string>

#include "ObjectFuncDefinitions.h"

class CudaGPU;

class CPUMediumGroupI;
class CPUTransformGroupI;
class CPUCameraGroupI;
class CPULightGroupI;
class CPUEndpointGroupI;

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
using CPUTransformGPtr = SharedLibPtr<CPUTransformGroupI>;
using CPUMediumGPtr = SharedLibPtr<CPUMediumGroupI>;
using CPUCameraGPtr = SharedLibPtr<CPUCameraGroupI>;
using CPULightGPtr = SharedLibPtr<CPULightGroupI>;

template <class T>
using NamedList = std::map<std::string, T>;

// Kernel Mappings
using AcceleratorBatchMap = std::map<uint32_t, GPUAcceleratorGroupI*>;
using WorkBatchArray = std::vector<GPUWorkBatchI*>;
using WorkBatchMap = std::map<uint32_t, WorkBatchArray>;
using WorkBatchCreationInfo = std::vector<std::tuple<uint32_t,
                                                     const GPUPrimitiveGroupI*,
                                                     const GPUMaterialGroupI*>>;
using BoundaryWorkBatchCreationInfo = std::vector<const CPUEndpointGroupI*>;

// Logic Independent parameters for tracer
// Logic Dependent ones will be provided by TracerOptionsI
struct TracerParameters
{    
    bool        verbose;    // Let Tracer to Log what it is doing   
    uint32_t    seed;       // RNG Seed
};