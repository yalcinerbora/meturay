#include "RayLib/TracerOptions.h"
#include "RayLib/TracerCallbacksI.h"

#include "DirectTracer.h"
#include "PathTracer.h"
#include "AOTracer.h"
#include "PPGTracer.h"

// Variant Does not compile on cuda code
// special cpp for functions that uses "TracerOptions"
void PathTracer::AskOptions()
{
    // Generate Tracer Object
    VariableList list;
    list.emplace(SAMPLE_NAME, OptionVariable(options.sampleCount));
    list.emplace(MAX_DEPTH_NAME, OptionVariable(options.maximumDepth));
    list.emplace(NEE_NAME, OptionVariable(options.nextEventEstimation));
    list.emplace(DIRECT_LIGHT_MIS_NAME, OptionVariable(options.directLightMIS));
    list.emplace(RR_START_NAME, OptionVariable(options.rrStart));

    std::string lightSamplerTypeString = LightSamplerTypeToString(options.lightSamplerType);
    list.emplace(LIGHT_SAMPLER_TYPE_NAME, OptionVariable(lightSamplerTypeString));

    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}

void DirectTracer::AskOptions()
{
    // Generate Tracer Object
    VariableList list;
    list.emplace(SAMPLE_NAME, OptionVariable(options.sampleCount));
    std::string renderTypeString = RenderTypeToString(options.renderType);
    list.emplace(RENDER_TYPE_NAME, OptionVariable(renderTypeString));

    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}

void AOTracer::AskOptions()
{
    // Generate Tracer Object
    VariableList list;
    list.emplace(SAMPLE_NAME, OptionVariable(options.sampleCount));

    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}

void PPGTracer::AskOptions()
{
    // Generate Tracer Object
    VariableList list;
    list.emplace(SAMPLE_NAME, OptionVariable(options.sampleCount));
    list.emplace(MAX_DEPTH_NAME, OptionVariable(options.maximumDepth));
    list.emplace(NEE_NAME, OptionVariable(options.nextEventEstimation));

    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}