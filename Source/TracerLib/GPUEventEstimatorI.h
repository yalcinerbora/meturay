#pragma once

#include <vector>

#include "RayLib/Vector.h"
#include "NodeListing.h"

struct SceneError;

class GPUEventEstimatorI
{
    public:
        virtual                 ~GPUEventEstimatorI() = default;

        // Interface
        virtual SceneError      Initialize(const NodeListing& lightList,
                                           // Material Keys
                                           const MaterialKeyListing& hitKeys,
                                           const std::map<uint32_t, GPUPrimitiveGroupI>&) = 0;

        virtual SceneError      ConstructEventEstimator() = 0;
};
