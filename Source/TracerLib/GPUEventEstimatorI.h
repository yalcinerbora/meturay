#pragma once

#include <vector>

#include "RayLib/Vector.h"
#include "NodeListing.h"

struct SceneError;

class CudaSystem;

class GPUEventEstimatorI
{
    public:
        virtual                 ~GPUEventEstimatorI() = default;

        // Interface
        // Type (as string) of the primitive group
        virtual const char* Type() const = 0;
        // Fetches Data for Nodes
        virtual SceneError      Initialize(const NodeListing& lightList,
                                           // Material Keys
                                           const MaterialKeyListing& hitKeys,
                                           const std::map<uint32_t, GPUPrimitiveGroupI>&) = 0;
        // Constructs Event Estimator
        virtual SceneError      ConstructEventEstimator(const CudaSystem&) = 0;
};