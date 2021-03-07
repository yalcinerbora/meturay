
// Common
#include "RayLib/System.h"
#include "RayLib/Log.h"
#include "RayLib/UTF8StringConversion.h"
//#include "RayLib/Constants.h"
//#include "RayLib/SurfaceLoaderGenerator.h"
//#include "RayLib/TracerStatus.h"
//#include "RayLib/DLLError.h"
//#include "RayLib/GPUTracerI.h"

//// Visor
//#include "RayLib/VisorI.h"
//#include "RayLib/VisorWindowInput.h"
//#include "RayLib/MovementSchemes.h"
//#include "VisorGL/VisorGLEntry.h"
//
//// Tracer
////#include "TracerLib/GPUSceneJson.h"
////#include "TracerLib/ScenePartitioner.h"
////#include "TracerLib/TracerLogicGenerator.h"

// Node
#include "RayLib/SelfNode.h"
//#include "RayLib/VisorCallbacksI.h"
//#include "RayLib/TracerCallbacksI.h"
//#include "RayLib/NodeI.h"
//#include "RayLib/AnalyticData.h"
//#include "RayLib/TracerError.h"
//#include "RayLib/TracerOptions.h"

#include <CLI11.hpp>
#include <array>

int main(int argc, const char* argv[])
{
    // Fancy CMD
    EnableVTMode();

    std::array<int, 2> resolution;
    std::string tracerConfig;
    std::string visorConfig;
    std::string sceneFileName = "";

    // Header
    const std::string BundleName = ProgramConstants::ProgramName;
    const std::string AppName = "MRay";
    const std::string Description = "Single Platform CPU Renderer and Visualizer";
    const std::string header = (BundleName + " - " + AppName + " " + Description);

    // Command Line Arguments
    CLI::App app{header};
    app.footer(ProgramConstants::Footer);
    app.add_option("-t,--tracerConfig", tracerConfig, "Tracer Configuration json File")
        ->required()
        ->expected(1)
        ->check(CLI::ExistingFile);
    app.add_option("-v,--visorConfig", visorConfig, "Visor Configuration json File")
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
        METU_LOG(app.help().c_str());
    else try 
    {
        app.parse((argc), (argv));
    }
    catch(const CLI::ParseError& e) 
    {
        return (app).exit(e);
    }

    std::cout << "Res   " << resolution[0] << "x" << resolution[1] << std::endl;
    std::cout << "TC    " << tracerConfig << std::endl;
    std::cout << "VC    " << visorConfig << std::endl;
    std::cout << "Scene " << sceneFileName << std::endl;


    // Parse Tracer Config
    TracerOptions tracerOptions;
    TracerParameters tracerParameters;
    std::string tracerType;

    // Parse Visor Config
    VisorOptions visorOpts;
    std::string visorDLL;
    MovementScemeList MovementSchemeList = {};
    KeyboardKeyBindings KeyBinds = VisorConstants::DefaultKeyBinds;
    MouseKeyBindings MouseBinds = VisorConstants::DefaultButtonBinds;
    visorInput = std::make_unique<VisorWindowInput>(std::move(KeyBinds),
                                                    std::move(MouseBinds),
                                                    std::move(MovementSchemeList),
                                                    VisorCamera{});

    // Generate Visor
    SharedLibPtr<VisorI> visor;

    visor->SetInputScheme(*visorInput);

    // Generate Tracer
    SharedLibPtr<GPUTracerI> tracer;
    TracerError tError = tracerSystem->GenerateTracer(tracer,
                                                      tracerParameters,
                                                      tracerOptions,
                                                      tracerType);
    ERROR_CHECK(TracerError, tError);

    // Create a Self Node
    SelfNode selfNode(*visor, *tracer);


    NodeError nError = selfNode.Initialize();
    ERROR_CHECK(NodeError, nError);




    // Do work loop of the self node
    try
    {
        // If scene file is provided as a argument set scene for the node        
        selfNode.ChangeScene(Utility::CopyStringU8(sceneFileName));
        if(!sceneFileName.empty())

        // Work returns when a crash occurs or user terminates
        selfNode.Work();
    }
    catch(TracerException const& e)
    {
        std::string err = static_cast<TracerError>(e);
        METU_ERROR_LOG("%s (%s)", err.c_str(), e.what());
        return 1;
    }
    return 0;
}