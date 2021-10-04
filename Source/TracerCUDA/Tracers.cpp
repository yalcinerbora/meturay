#include "RayLib/TracerOptions.h"
#include "RayLib/TracerCallbacksI.h"

#include "Tracers.h"
#include "DirectTracer.h"
#include "PathTracer.h"
#include "AOTracer.h"
#include "PPGTracer.h"
#include "RefPGTracer.h"

const std::array<std::string, static_cast<size_t>(LightSamplerType::END)> SamplerNames =
{
    "Uniform"
};

TracerError LightSamplerCommon::StringToLightSamplerType(LightSamplerType& ls,
                                                         const std::string& lsName)
{
    uint32_t i = 0;
    for(const std::string s : SamplerNames)
    {
        if(lsName == s)
        {
            ls = static_cast<LightSamplerType>(i);
            return TracerError::OK;
        }
        i++;
    }
    return TracerError::UNABLE_TO_INITIALIZE;
}

std::string LightSamplerCommon::LightSamplerTypeToString(LightSamplerType t)
{
    return SamplerNames[static_cast<int>(t)];
}

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

    std::string lightSamplerTypeString = LightSamplerCommon::LightSamplerTypeToString(options.lightSamplerType);
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

void RefPGTracer::AskOptions()
{
    // Generate Tracer Object
    VariableList list;
    list.emplace(SAMPLE_NAME, OptionVariable(options.samplePerIteration));
    list.emplace(MAX_DEPTH_NAME, OptionVariable(options.maximumDepth));
    list.emplace(NEE_NAME, OptionVariable(options.nextEventEstimation));

    if(callbacks) callbacks->SendCurrentOptions(TracerOptions(std::move(list)));
}