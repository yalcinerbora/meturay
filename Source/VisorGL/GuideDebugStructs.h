#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>

#include "RayLib/StripComments.h"
#include "RayLib/SceneIO.h"

using GuiderConfigs = std::vector<std::pair<std::string, nlohmann::json>>;

struct GuideDebugConfig
{
    std::string             sceneName;
    std::string             refImage;
    std::string             posImage;
    uint32_t                depthCount;

    std::vector<Vector3f>   gradientValues;

    nlohmann::json          refGuideConfig;

    GuiderConfigs           guiderConfigs;
};

namespace GuideDebug
{
    //static constexpr const char* NAME = "name";
    static constexpr const char* TYPE = "type";

    static constexpr const char* DATA_GRADIENT_NAME = "DataGradient";
    static constexpr const char* SCENE_NAME = "Scene";
    static constexpr const char* SCENE_IMAGE = "refImage";
    static constexpr const char* SCENE_POS_IMAGE = "posImage";
    static constexpr const char* SCENE_DEPTH = "depth";
    static constexpr const char* NAME = "name";

    static constexpr const char* PG_NAME = "PathGuiders";
    static constexpr const char* REFERENCE_NAME = "Reference";

    bool ParseConfigFile(GuideDebugConfig&, const std::u8string& fileName);
}

inline bool GuideDebug::ParseConfigFile(GuideDebugConfig& s, const std::u8string& fileName)
{
    // Always assume filenames are UTF-8
    const auto path = std::filesystem::path(fileName);
    std::ifstream file(path);

    if(!file.is_open()) return false;
    // Parse Json
    nlohmann::json jsonFile = nlohmann::json::parse(file, nullptr,
                                                    true, true);

    s.sceneName = jsonFile[SCENE_NAME][NAME];
    s.refImage = jsonFile[SCENE_NAME][SCENE_IMAGE];
    s.posImage = jsonFile[SCENE_NAME][SCENE_POS_IMAGE];
    s.depthCount = jsonFile[SCENE_NAME][SCENE_DEPTH];

    // Reference Config
    s.refGuideConfig = jsonFile[REFERENCE_NAME];

    for(const nlohmann::json& j : jsonFile[PG_NAME])
    {
        s.guiderConfigs.emplace_back(j[TYPE], j);
    }

    // Load Gradient Values
    for(const nlohmann::json& j : jsonFile[DATA_GRADIENT_NAME])
    {
        s.gradientValues.emplace_back(SceneIO::LoadVector<3,float>(j));
    }
    return true;
}