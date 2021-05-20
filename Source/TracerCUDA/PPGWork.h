#pragma once

#include <cstdint>

struct PathGuidingOptions
{
    // Parameter for determining the spatial subdivision occurance
    uint32_t    sTreeSplitThreshold;
    // Parameter for determining the spatical subdivison occurance
    float       dTreeSplitThreshold;
};