#include "ConfigParsers.h"
#include "Log.h"
#include "SceneIO.h"
#include "SharedLib.h"
#include "TracerOptions.h"
#include "StripComments.h"
#include "EnumStringConversions.h"
#include "MovementSchemes.h"

#include "TracerSystemI.h"
#include "VisorI.h"

#include <nlohmann/json.hpp>
#include <filesystem>

namespace ConfigParser
{
    void ParseDLL(SharedLibArgs& entryExit, std::string& dllName,
                  const nlohmann::json& jsonObj)
    {
        static constexpr const char* DLL_NAME = "name";
        static constexpr const char* DLL_ENTRY = "entry";
        static constexpr const char* DLL_EXIT = "exit";

        entryExit.mangledConstructorName = SceneIO::LoadString(jsonObj[DLL_ENTRY]);
        entryExit.mangledDestructorName = SceneIO::LoadString(jsonObj[DLL_EXIT]);
        dllName = SceneIO::LoadString(jsonObj[DLL_NAME]);
    }

    void ParseTracerParameters(TracerParameters& params,
                               const nlohmann::json& paramsJson)
    {
        static constexpr const char* SEED_NAME = "seed";
        static constexpr const char* VERBOSE_NAME = "verbose";
        static constexpr const char* FORCE_OPTIX_NAME = "forceOptiX";

        params.seed = SceneIO::LoadNumber<uint32_t>(paramsJson[SEED_NAME]);
        params.verbose = SceneIO::LoadBool(paramsJson[VERBOSE_NAME]);
        params.forceOptiX = SceneIO::LoadBool(paramsJson[FORCE_OPTIX_NAME]);
    }

    void ParseVisorOptions(VisorOptions& opts,
                           const nlohmann::json& optsJson)
    {
        static constexpr const char* EVENT_BUFFER_SIZE_NAME = "eventBufferSize";
        static constexpr const char* STEREO_ON_NAME = "stereoRendering";
        //static constexpr const char* VSYNC_ON_NAME = "vSync";
        static constexpr const char* W_PIXEL_FORMAT_NAME = "windowPixelFormat";
        static constexpr const char* WINDOW_SIZE_NAME = "windowSize";
        static constexpr const char* FPS_LIMIT_NAME = "fpsLimit";
        static constexpr const char* ENABLE_GUI_NAME = "enableGUI";
        static constexpr const char* ENABLE_TMO_NAME = "enableTMO";

        // Load VisorOptions
        opts.eventBufferSize = SceneIO::LoadNumber<uint32_t>(optsJson[EVENT_BUFFER_SIZE_NAME]);
        opts.stereoOn = SceneIO::LoadBool(optsJson[STEREO_ON_NAME]);
        opts.wFormat = EnumStringConverter::StringToPixelFormatType(optsJson[W_PIXEL_FORMAT_NAME]);
        opts.wSize = SceneIO::LoadVector<2, int>(optsJson[WINDOW_SIZE_NAME]);
        opts.fpsLimit = SceneIO::LoadNumber<float>(optsJson[FPS_LIMIT_NAME]);
        opts.enableGUI = SceneIO::LoadBool(optsJson[ENABLE_GUI_NAME]);
        opts.enableTMO = SceneIO::LoadBool(optsJson[ENABLE_TMO_NAME]);
    }
}

bool ConfigParser::ParseVisorOptions(// Visor Input Related
                                     KeyboardKeyBindings& keyBindings,
                                     MouseKeyBindings& mouseBindings,
                                     MovementSchemeList& movementSchemes,
                                     // Visor Related
                                     VisorOptions& opts,
                                     // Visor DLL Related
                                     std::string& visorDLLName,
                                     SharedLibArgs& dllEntryPointName,
                                     //
                                     const std::string& configFileName)
{
    static constexpr const char* KEY_BIND_NAME = "KeyBindings";
    static constexpr const char* MOUSE_BIND_NAME = "MouseBindings";
    static constexpr const char* MOVEMENT_SCEHEMES_NAME = "MovementScehemes";
    static constexpr const char* VISOR_OPTIONS_NAME = "VisorOptions";
    static constexpr const char* VISOR_DLL_NAME = "VisorDLL";

    try
    {
        // Always assume filenames are UTF-8
        const auto path = std::filesystem::path(configFileName);
        std::ifstream file(path);

        if(!file.is_open()) return SceneError::FILE_NOT_FOUND;
        // Parse Json
        nlohmann::json configJson = nlohmann::json::parse(file, nullptr,
                                                          true, true);
        // Json is Loaded
        // Load Key Binds
        auto keyJson = configJson.end();
        if((keyJson = configJson.find(KEY_BIND_NAME)) != configJson.end())
        {
            // TODO: Implement
            return false;
        }
        else keyBindings = VisorConstants::DefaultKeyBinds;
        // Load Button Binds
        auto buttonJson = configJson.end();
        if((buttonJson = configJson.find(MOUSE_BIND_NAME)) != configJson.end())
        {
            // TODO: Implement
            return false;
        }
        else mouseBindings = VisorConstants::DefaultButtonBinds;

        // Load Movement Schemes
        movementSchemes.clear();
        auto schemesJson = configJson.end();
        if((schemesJson = configJson.find(MOVEMENT_SCEHEMES_NAME)) != configJson.end())
        {
            //movementSchemes.reserve(schemesJson->size());
            // TODO: Implement
            return false;
        }
        else
        {
            // Add defaults
            movementSchemes.emplace_back(new MovementSchemeFPS());
            movementSchemes.emplace_back(new MovementSchemeMaya());
        }

        // Load VisorOptions
        ParseVisorOptions(opts, configJson[VISOR_OPTIONS_NAME]);

        // Load Visor DLL
        ParseDLL(dllEntryPointName, visorDLLName,
                 configJson[VISOR_DLL_NAME]);
    }
    catch(nlohmann::json::parse_error const& e)
    {
        std::string errAsString(e.what());

        METU_ERROR_LOG("{:s}", errAsString);
        return false;
    }
    return true;
}

bool ConfigParser::ParseTracerOptions(// Tracer Related
                                      TracerOptions& tracerOptions,
                                      TracerParameters& tracerParameters,
                                      std::string& tracerTypeName,
                                      // Tracer DLL Related
                                      std::string& tracerDLLName,
                                      SharedLibArgs& dllEntryPointName,
                                      // Misc
                                      std::vector<SurfaceLoaderSharedLib>& surfaceLoaderDLLs,
                                      ScenePartitionerType& gpuUsage,
                                      //
                                      const std::string& configFileName)
{
    static constexpr const char* TRACER_OPTS_NAME = "TracerOptions";
    static constexpr const char* TRACER_PARAMS_NAME = "TracerParameters";
    static constexpr const char* TRACER_DLL_NAME = "TracerDLL";
    static constexpr const char* SURFACE_LOADERS_NAME = "SurfaceLoaders";

    static constexpr const char* TRACER_TYPE_NAME = "TracerTypeName";
    static constexpr const char* PARTITION_TYPE_NAME = "ScenePartitionType";

    auto ParseVariantObject = [](std::string& name, OptionVariable& var,
                                 const nlohmann::json& jsnObj) -> bool
    {
        static constexpr const char* VARIANT_NAME = "name";
        static constexpr const char* VARIANT_TYPE_NAME = "type";
        static constexpr const char* VARIANT_VAL_NAME = "val";

        name = SceneIO::LoadString(jsnObj[VARIANT_NAME]);
        std::string typeName = SceneIO::LoadString(jsnObj[VARIANT_TYPE_NAME]);
        TracerOptionsI::OptionType optType = EnumStringConverter::StringToTracerOptionType(typeName);
        const nlohmann::json& valJson = jsnObj[VARIANT_VAL_NAME];

        // TODO: Better way to do this?
        switch(optType)
        {
            case TracerOptionsI::BOOL:
                var = SceneIO::LoadBool(valJson);
                break;
            case TracerOptionsI::INT32:
                var = SceneIO::LoadNumber<int32_t>(valJson);
                break;
            case TracerOptionsI::UINT32:
                var = SceneIO::LoadNumber<uint32_t>(valJson);
                break;
            case TracerOptionsI::FLOAT:
                var = SceneIO::LoadNumber<float>(valJson);
                break;
            case TracerOptionsI::VECTOR2I:
                var = SceneIO::LoadVector<2, int32_t>(valJson);
                break;
            case TracerOptionsI::VECTOR2UI:
                var = SceneIO::LoadVector<2, uint32_t>(valJson);
                break;
            case TracerOptionsI::VECTOR2:
                var = SceneIO::LoadVector<2, float>(valJson);
                break;
            case TracerOptionsI::VECTOR3:
                var = SceneIO::LoadVector<3, float>(valJson);
                break;
            case TracerOptionsI::VECTOR4:
                var = SceneIO::LoadVector<4, float>(valJson);
                break;
            case TracerOptionsI::STRING:
                var = SceneIO::LoadString(valJson);
                break;
            default:
                return false;
        }
        return true;
    };

    try
    {
        // Always assume filenames are UTF-8
        const auto path = std::filesystem::path(configFileName);
        std::ifstream file(path);

        if(!file.is_open()) return SceneError::FILE_NOT_FOUND;
        // Parse Json
        nlohmann::json configJson = nlohmann::json::parse(file, nullptr,
                                                          true, true);
        // Json is Loaded

        // Load Tracer Options
        VariableList varList;
        const auto& tOptsJson = configJson[TRACER_OPTS_NAME];
        for(const auto& optsNode : tOptsJson)
        {
            OptionVariable var;
            std::string varName;

            if(!ParseVariantObject(varName, var, optsNode))
                return false;

            varList.emplace(varName, var);
        }
        tracerOptions = TracerOptions(std::move(varList));

        // Load Tracer Parameters
        ParseTracerParameters(tracerParameters, configJson[TRACER_PARAMS_NAME]);

        // Load Tracer DLL
        ParseDLL(dllEntryPointName, tracerDLLName,
                 configJson[TRACER_DLL_NAME]);

        // Load Surface Loader DLLs
        surfaceLoaderDLLs.clear();
        const auto& sLoadersJson = configJson[SURFACE_LOADERS_NAME];
        surfaceLoaderDLLs.reserve(sLoadersJson.size());
        for(const auto& sLoaderJson : sLoadersJson)
        {
            static constexpr const char* REGEX_NAME = "regex";

            SurfaceLoaderSharedLib sLib;
            ParseDLL(sLib.mangledName, sLib.libName, sLoaderJson);
            sLib.regex = SceneIO::LoadString(sLoaderJson[REGEX_NAME]);
            surfaceLoaderDLLs.emplace_back(sLib);
        }
        tracerTypeName = SceneIO::LoadString(configJson[TRACER_TYPE_NAME]);

        gpuUsage = EnumStringConverter::StringToScenePartitionerType(SceneIO::LoadString(configJson[PARTITION_TYPE_NAME]));
        // All done!
    }
    catch(nlohmann::json::exception const& e)
    {
        METU_ERROR_LOG("{:s}", std::string(e.what()));
        return false;
    }
    return true;
}