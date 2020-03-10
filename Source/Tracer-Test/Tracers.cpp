#include "RayLib/TracerOptions.h"
#include "RayLib/TracerCallbacksI.h"

#include "BasicTracer.cuh"
#include "PathTracer.cuh"

// Variant Does not compile on cuda code
// special cpp for functions that uses "TracerOptions"

void BasicTracer::AskOptions()
{
    // Generate Tracer Object
    VariableList list;
    list.emplace(SAMPLE_NAME, OptionVariable(options.sampleCount));

    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}