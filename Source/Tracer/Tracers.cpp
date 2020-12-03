#include "RayLib/TracerOptions.h"
#include "RayLib/TracerCallbacksI.h"

#include "DirectTracer.h"
#include "PathTracer.h"

// Variant Does not compile on cuda code
// special cpp for functions that uses "TracerOptions"
void PathTracer::AskOptions()
{
    // Generate Tracer Object
    VariableList list;
    list.emplace(SAMPLE_NAME, OptionVariable(options.sampleCount));
    list.emplace(MAX_DEPTH_NAME, OptionVariable(options.maximumDepth));
    list.emplace(NEE_NAME, OptionVariable(options.nextEventEstimation));

    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}

void DirectTracer::AskOptions()
{
    // Generate Tracer Object
    VariableList list;
    list.emplace(SAMPLE_NAME, OptionVariable(options.sampleCount));

    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}