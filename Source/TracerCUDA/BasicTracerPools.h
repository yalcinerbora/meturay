#pragma once

#include "TracerCUDALib/TracerLogicGenerator.h"

class BasicTracerPool final : public TracerPoolI
{
    public:
        // Constructors & Destructor
        BasicTracerPool();
        ~BasicTracerPool() = default;
};