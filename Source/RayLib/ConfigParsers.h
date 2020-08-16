#pragma once

#include <string>

#include "SharedLib.h"
#include "VisorInputStructs.h"
#include "VisorI.h"

using LibInfo = std::tuple<std::string, SharedLibArgs, std::string>;

namespace ConfigParser
{
    //bool ParseVisorOptions(KeyboardKeyBindings& keyBindings,
    //                       MouseKeyBindings& mouseBindings,
    //                       VisorOptions& opts,
    //                       const std::u8string& file);
    //bool ParseTracerOptions(std::vector<LibInfo>& tracerPools,
    //                        std::vector<LibInfo>& materialPools,
    //                        std::vector<LibInfo>& primitivePools,
    //                        std::vector<LibInfo>& acceleratorPools,
    //                        std::vector<LibInfo>& tracerPooDLLs,
    //                        std::vector<LibInfo>& surfaceLoaderDLLs);
}