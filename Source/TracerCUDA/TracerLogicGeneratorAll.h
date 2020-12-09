#pragma once

#include "TracerCUDALib/TracerLogicGenerator.h"

class TracerLogicGeneratorAll : public TracerLogicGenerator
{
    public:
        // Constructors & Destructor
                                    TracerLogicGeneratorAll();
                                    TracerLogicGeneratorAll(const TracerLogicGeneratorAll&) = delete;
        TracerLogicGeneratorAll&    operator=(const TracerLogicGeneratorAll&) = delete;
                                        ~TracerLogicGeneratorAll() = default;
};
