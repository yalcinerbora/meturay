
// Common
#include "RayLib/System.h"
#include "RayLib/Log.h"
#include "RayLib/UTF8StringConversion.h"
#include "RayLib/DLLError.h"
#include "RayLib/SharedLib.h"
#include "RayLib/ConfigParsers.h"
// Visor
#include "RayLib/VisorI.h"
#include "RayLib/VisorWindowInput.h"
#include "RayLib/MovementSchemes.h"
// Tracer
#include "RayLib/TracerOptions.h"
#include "RayLib/TracerSystemI.h"
// Node
#include "RayLib/SelfNode.h"
// Args Parser
#include <CLI11.hpp>

#include <array>

int main(int argc, const char* argv[])
{
    // Fancy CMD
    EnableVTMode();

    std::array<int, 2> resolution;
    std::string tracerConfigFileName;
    std::string visorConfigFileName;
    std::string sceneFileName = "";

    // Header
    const std::string BundleName = ProgramConstants::ProgramName;
    const std::string AppName = "MRay";
    const std::string Description = "Single Platform CPU Renderer and Visualizer";
    const std::string header = (BundleName + " - " + AppName + " " + Description);

    // Command Line Arguments
    CLI::App app{header};
    app.footer(ProgramConstants::Footer);
    app.add_option("-t,--tracerConfig", tracerConfigFileName, "Tracer Configuration json File")
        ->required()
        ->expected(1)
        ->check(CLI::ExistingFile);
    app.add_option("-v,--visorConfig", visorConfigFileName, "Visor Configuration json File")
        ->required()
        ->expected(1)
        ->check(CLI::ExistingFile);   
    app.add_option("-r, --resolution", resolution, "Initial Image Resolution")
        ->required()
        ->expected(1)
        ->check(CLI::Number)
        ->delimiter('x');
    app.add_option("Scene File", sceneFileName, "Scene file");

    if(argc == 1)
    {
        METU_LOG(app.help().c_str());
        return 0;
    }

    try 
    {
        app.parse((argc), (argv));
    }
    catch(const CLI::ParseError& e) 
    {
        return (app).exit(e);
    }

    //std::cout << "Res   " << resolution[0] << "x" << resolution[1] << std::endl;
    //std::cout << "TC    " << tracerConfigFileName << std::endl;
    //std::cout << "VC    " << visorConfigFileName << std::endl;
    //std::cout << "Scene " << sceneFileName << std::endl;

    // Error Vars
    TracerError tError = TracerError::OK;
    DLLError dError = DLLError::OK;
    NodeError nError = NodeError::OK;

    // Variables Related to the MRay
    SharedLibPtr<TracerSystemI> tracerSystem = {nullptr, nullptr};
    SharedLibPtr<GPUTracerI> tracer = {nullptr, nullptr};
    std::unique_ptr<VisorInputI> visorInput = nullptr;

    // Parse Tracer Config
    TracerOptions tracerOptions;
    TracerParameters tracerParameters;
    SharedLibArgs tracerDLLEntryFunctionNames;
    std::string tracerDLLName;
    std::string tracerTypeName;
    std::vector<SurfaceLoaderSharedLib> surfaceLoaderLibraries;
    ScenePartitionerType scenePartitionType;

    if(!ConfigParser::ParseTracerOptions(tracerOptions, tracerParameters,
                                         tracerTypeName, tracerDLLName,
                                         tracerDLLEntryFunctionNames,
                                         surfaceLoaderLibraries, scenePartitionType,
                                         //
                                         tracerConfigFileName))
    {
        METU_ERROR_LOG("Unable to parse Tracer Options");
        return 1;
    }

    // Parse Visor Config
    VisorOptions visorOpts;
    std::string visorDLLName;
    SharedLibArgs visorDLLEntryFunctionNames;
    MovementScemeList movementSchemeList = {};
    VisorCamera visorCamera;
    KeyboardKeyBindings keyBinds;
    MouseKeyBindings mouseBinds;

    if(!ConfigParser::ParseVisorOptions(keyBinds, mouseBinds,
                                        movementSchemeList, visorCamera,
                                        visorOpts, visorDLLName,
                                        visorDLLEntryFunctionNames,
                                        //
                                        visorConfigFileName))
    {
        METU_ERROR_LOG("Unable to parse Visor Options");
        return 1;
    }

    // Generate Visor
    SharedLib visorDLL(visorDLLName);
    SharedLibPtr<VisorI> visor = {nullptr, nullptr};
    dError = visorDLL.GenerateObjectWithArgs(visor, visorDLLEntryFunctionNames,
                                             // Args
                                             visorOpts,
                                             resolution,
                                             PixelFormat::RGBA_FLOAT);
    ERROR_CHECK(DLLError, dError);
    visorInput = std::make_unique<VisorWindowInput>(std::move(keyBinds),
                                                    std::move(mouseBinds),
                                                    std::move(movementSchemeList),
                                                    visorCamera);
    visor->SetInputScheme(*visorInput);

    // Generate Tracer
    SharedLib tracerDLL(tracerDLLName);    
    dError = tracerDLL.GenerateObject(tracerSystem, tracerDLLEntryFunctionNames);
    ERROR_CHECK(DLLError, dError);
    tError = tracerSystem->Initialize(surfaceLoaderLibraries, scenePartitionType);
    ERROR_CHECK(TracerError, tError);

    // Create a Self Node
    SelfNode selfNode(*visor, *tracerSystem,
                      tracerOptions, tracerParameters,
                      tracerTypeName,
                      Vector2i(resolution.data()));
    nError = selfNode.Initialize();
    visorInput->AttachVisorCallback(selfNode);
    ERROR_CHECK(NodeError, nError);

    // Do work loop of the self node
    try
    {
        // If scene file is provided as a argument set scene for the node
        if(!sceneFileName.empty())
            selfNode.ChangeScene(Utility::CopyStringU8(sceneFileName));

        // Work returns when a crash occurs or user terminates
        selfNode.Work();
    }
    catch(TracerException const& e)
    {
        std::string err = static_cast<TracerError>(e);
        METU_ERROR_LOG("%s (%s)", err.c_str(), e.what());
        return 1;
    }

    // Orderly Delete Unique Ptrs
    // Shared lib dependent ptrs should be released first
    tracerSystem = SharedLibPtr<TracerSystemI>(nullptr, nullptr);
    visor = SharedLibPtr<VisorI>(nullptr, nullptr);
    // Shared Libs etc. will destroy on scope exit
    return 0;
}