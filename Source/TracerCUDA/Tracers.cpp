
#include "Tracers.h"
#include <array>

const std::array<std::string, static_cast<size_t>(LightSamplerType::END)> SamplerNames =
{
    "Uniform"
};

TracerError LightSamplerCommon::StringToLightSamplerType(LightSamplerType& ls,
                                                         const std::string& lsName)
{
    uint32_t i = 0;
    for(const std::string& s : SamplerNames)
    {
        if(lsName == s)
        {
            ls = static_cast<LightSamplerType>(i);
            return TracerError::OK;
        }
        i++;
    }
    return TracerError::UNABLE_TO_INITIALIZE_TRACER;
}

std::string LightSamplerCommon::LightSamplerTypeToString(LightSamplerType t)
{
    return SamplerNames[static_cast<int>(t)];
}