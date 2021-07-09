#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include "RayLib/StripComments.h"

using GuiderConfigs = std::map<std::string, nlohmann::json>;

struct GuideDebugConfig
{
    std::string     refImage;
    uint32_t        depthCount;
    GuiderConfigs   guiderConfigs;
};

namespace GuideDebug
{
    //static constexpr const char* NAME = "name";
    static constexpr const char* TYPE = "type";

    static constexpr const char* SCENE_NAME = "Scene";
    static constexpr const char* SCENE_IMAGE = "img";
    static constexpr const char* SCENE_DEPTH = "depth";

    static constexpr const char* PG_NAME = "PathGuiders";

    bool ParseConfigFile(GuideDebugConfig&, const std::u8string& fileName);
}

inline bool GuideDebug::ParseConfigFile(GuideDebugConfig& s, const std::u8string& fileName)
{
    // Always assume filenames are UTF-8
    const auto path = std::filesystem::path(fileName);
    std::ifstream file(path);

    if(!file.is_open()) return false;
    auto stream = Utility::StripComments(file);

    // Parse Json
    nlohmann::json jsonFile;
    stream >> (jsonFile);

    s.refImage = jsonFile[SCENE_NAME][SCENE_IMAGE];
    s.depthCount = jsonFile[SCENE_NAME][SCENE_DEPTH];
    
    for(const nlohmann::json& j : jsonFile[PG_NAME])
    {
        s.guiderConfigs.emplace(j[TYPE], j);
    }        
    return true;
}