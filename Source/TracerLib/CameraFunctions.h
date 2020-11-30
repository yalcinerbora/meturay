#pragma once

#include "RayMemory.h"

// Commands that initialize ray auxiliary data
template<class RayAuxData>
using AuxInitFunc = void(*)(RayAuxData&,
                            // Input
                            const RayAuxData&,
                            const RayReg&,
                            // Index
                            const uint16_t mediumIndex,
                            const uint32_t localPixelId,
                            const uint32_t pixelSampleId);