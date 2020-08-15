
// Common
#include "RayLib/System.h"
#include "RayLib/Log.h"
#include "RayLib/Constants.h"
#include "RayLib/SurfaceLoaderGenerator.h"
#include "RayLib/TracerStatus.h"
#include "RayLib/DLLError.h"
#include "RayLib/GPUTracerI.h"

// Visor
#include "RayLib/VisorI.h"
#include "RayLib/VisorWindowInput.h"
#include "RayLib/MovementSchemes.h"
#include "VisorGL/VisorGLEntry.h"

// Tracer
//#include "TracerLib/GPUSceneJson.h"
//#include "TracerLib/ScenePartitioner.h"
//#include "TracerLib/TracerLogicGenerator.h"

// Node
#include "RayLib/VisorCallbacksI.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/NodeI.h"
#include "RayLib/AnalyticData.h"
#include "RayLib/TracerError.h"
#include "RayLib/TracerOptions.h"

#include <CLI11.hpp>
#include <array>

class SelfNode
    : public VisorCallbacksI
    , public TracerCallbacksI
    , public NodeI
{
    // Visor is in MainThread
    VisorI& visor;
    GPUTracerI& tracer;
   /* GPUSceneI& scene;*/

//    std::thread


};

int main(int argc, const char* argv[])
{
    // Fancy CMD
    EnableVTMode();

    std::array<int, 2> resolution;
    std::string tracerConfig;
    std::string visorConfig;

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

    std::cout << "Res  " << resolution[0] << "x" << resolution[1] << std::endl;
    std::cout << "TC   " << tracerConfig << std::endl;
    std::cout << "VC   " << visorConfig << std::endl;

    return 0;
    //




    //// Meta Logic Pool of Materials, Accelerators, 
    //TracerLogicGenerator                generator;


    //SurfaceLoaderGenerator              surfaceLoaders;

    //std::unique_ptr<VisorI>             visorView;
    //std::unique_ptr<GPUSceneJson>       gpuScene;
    //std::unique_ptr<CudaSystem>         cudaSystem;

    //// Generate GPU List & A Partitioner
    //// Check cuda system error here
    //cudaSystem = std::make_unique<CudaSystem>();
    //CudaError cudaE = cudaSystem->Initialize();
    //ERROR_CHECK(CudaError, cudaE);

    //// GPU Data Partitioner
    //// Basically delegates all work to single GPU
    //SingleGPUScenePartitioner partitioner(*cudaSystem);

    //// Load Scene & get material/primitive mappings
    //gpuScene = std::make_unique<GPUSceneJson>(sceneName,
    //                                          partitioner,
    //                                          generator,
    //                                          surfaceLoaders,
    //                                          *cudaSystem);
    //SceneError scnE = gpuScene->LoadScene(sceneTime);
    //ERROR_CHECK(SceneError, scnE);



    //MovementScemeList MovementSchemeList = {};    
    //KeyboardKeyBindings KeyBinds = VisorConstants::DefaultKeyBinds;
    //MouseKeyBindings MouseBinds = VisorConstants::DefaultButtonBinds;

    //// Visor Input Generation
    //visorInput = std::make_unique<VisorWindowInput>(std::move(KeyBinds),
    //                                                std::move(MouseBinds),
    //                                                std::move(MovementSchemeList),
    //                                                CPUCamera{});
    //                                                
    //// Window Params
    //VisorOptions visorOpts;
    //visorOpts.stereoOn = false;
    //visorOpts.vSyncOn = false;
    //visorOpts.eventBufferSize = 128;
    //visorOpts.fpsLimit = 24.0f;

    //visorOpts.wSize = SCREEN_RESOLUTION;
    //visorOpts.wFormat = WINDOW_PIXEL_FORMAT;


    //visorOpts.iFormat = IMAGE_PIXEL_FORMAT;
    //visorOpts.iSize = MockNode::IMAGE_RESOLUTION;

    //// Create Visor
    //visorView = CreateVisorGL(visorOpts);
    //visorView->SetInputScheme(*visorInput);

    //// Set Window Res wrt to monitor resolution
    //Vector2i newImgSize = 3 * visorView->MonitorResolution() / 5;
    //float ratio = static_cast<float>(newImgSize[1]) / SCREEN_RESOLUTION[1];
    //newImgSize[0] = static_cast<int>(SCREEN_RESOLUTION[0] * ratio);
    //newImgSize[1] = static_cast<int>(SCREEN_RESOLUTION[1] * ratio);
    //visorView->SetWindowSize(newImgSize);


    //while(visor.IsOpen())
    //{
    //    // Run tracer
    //    tracer.GenerateWork(0);

    //    // Render
    //    while(tracer.Render());
    //    // Finalize (send image to visor)
    //    tracer.Finalize();
    //    //printf("\n----------------------------------\n");

    //    // Render scene window
    //    visor.Render();

    //    // Present Back Buffer
    //    visor.ProcessInputs();

    //    // Timing and Window Termination
    //    t.Lap();
    //    elapsed += t.Elapsed<CPUTimeSeconds>();

    //    double elapsedS = t.Elapsed<CPUTimeSeconds>();
    //    double rps = SAMPLE_COUNT * SAMPLE_COUNT * IMAGE_RESOLUTION[0] * IMAGE_RESOLUTION[1];
    //    rps *= (1.0 / elapsedS);
    //    rps /= 1'000'000.0;

    //    fprintf(stdout, "%c[2K", 27);
    //    fprintf(stdout, "Time: %fs Rps: %fM ray/s  Total: %fm\r",
    //            elapsedS, rps, (elapsed / 60.0));
    //    //if(elapsed >= Duration) break;
    //}
    //METU_LOG("\n");

}