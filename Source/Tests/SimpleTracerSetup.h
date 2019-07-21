#pragma once

#include <gtest/gtest.h>

// Common
#include "RayLib/System.h"
#include "RayLib/Log.h"
#include "RayLib/Constants.h"
#include "RayLib/CPUTimer.h"
#include "RayLib/SurfaceLoaderGenerator.h"
#include "RayLib/TracerStatus.h"

// Visor
#include "RayLib/VisorI.h"
#include "RayLib/VisorWindowInput.h"
#include "VisorGL/VisorGLEntry.h"

// Tracer
#include "TracerLib/GPUSceneJson.h"
#include "TracerLib/TracerLoader.h"
#include "TracerLib/TracerBase.h"
#include "TracerLib/ScenePartitioner.h"

// Node
#include "RayLib/SelfNode.h"
#include "RayLib/TracerError.h"

#define ERROR_CHECK(ErrType, e) \
if(e != ErrType::OK) \
{\
    METU_ERROR_LOG("%s", static_cast<std::string>(scnE).c_str()); \
    return false;\
}

class SimpleTracerSetup
{
    private:
        static constexpr Vector2i           IMAGE_RESOLUTION = {256, 256};
        static constexpr double             WINDOW_DURATION = 3.5;
        static constexpr PixelFormat        IMAGE_PIXEL_FORMAT = PixelFormat::RGBA_FLOAT;

        static constexpr const char*        TRACER_DLL = "Tracer-Test";
        static constexpr const char*        TRACER_DLL_GENERATOR = "GenerateBasicTracer";
        static constexpr const char*        TRACER_DLL_DELETER = "DeleteBasicTracer";

        static constexpr TracerParameters   TRACER_PARAMETERS = 
        {
            0   // Seed
        };
        static constexpr TracerOptions      TRACER_OPTIONS =
        {
            false   // Verbose
        };

        SurfaceLoaderGenerator              surfaceLoaders;

        // Scene Tracer and Visor
        std::unique_ptr<VisorI>             visorView;
        std::unique_ptr<TracerBase>         tracerBase;
        std::unique_ptr<GPUSceneJson>       gpuScene;

        // Loaded DLL
        std::unique_ptr<SharedLib>          sharedLib;
        LogicInterface                      tracerGenerator;

        // Visor Input
        std::unique_ptr<VisorWindowInput>   visorInput;

        // Self Node
        std::unique_ptr<SelfNode>           selfNode;

        const std::string                   sceneName;
        const double                        sceneTime;

    public:
        // Constructors & Destructor
                            SimpleTracerSetup(std::string sceneName, 
                                              double sceneTime);
                            SimpleTracerSetup() = default;

        bool                Init();
        void                Body();
};

class MockVisorCallback
{

};

class MockTracerCallback
{

};

SimpleTracerSetup::SimpleTracerSetup(std::string sceneName, double sceneTime)
    : sceneName(sceneName)
    , sceneTime(sceneTime)
    , visorView(nullptr)
    , tracerBase(nullptr)
    , gpuScene(nullptr)
    , sharedLib(nullptr)
    , tracerGenerator(nullptr, nullptr)
    , visorInput(nullptr)
    , selfNode(nullptr)
{}

bool SimpleTracerSetup::Init()
{
    TracerStatus status =
    {
        sceneName,
        0,

        0,

        CameraPerspective{},

        IMAGE_RESOLUTION,
        IMAGE_PIXEL_FORMAT,

        0.0,

        false,
        false,

        TRACER_OPTIONS
    };

    // Load Tracer Genrator from DLL
    sharedLib = std::make_unique<SharedLib>(TRACER_DLL);
    tracerGenerator = TracerLoader::LoadTracerLogic(*sharedLib,
                                                    TRACER_DLL_GENERATOR,
                                                    TRACER_DLL_DELETER);

    // Generate GPU List & A Partitioner
    // Check cuda system error here
    const std::vector<CudaGPU>& gpuList = CudaSystem::GPUList();
    if(CudaSystem::SystemStatus() != CudaSystem::OK) return false;
    const int leaderDevice = gpuList[0].DeviceId();

    // GPU Data Partitioner
    SingleGPUScenePartitioner partitioner(gpuList);

    // Load Scene
    gpuScene = std::make_unique<GPUSceneJson>(sceneName,
                                              partitioner,
                                              *tracerGenerator.get(),
                                              surfaceLoaders);
    SceneError scnE = gpuScene->LoadScene(sceneTime);
    ERROR_CHECK(SceneError, scnE);


    // Finally generate logic after successfull load
    TracerBaseLogicI* logic;
    scnE = tracerGenerator->GenerateBaseLogic(logic, TRACER_PARAMETERS,
                                              gpuScene->MaxMatIds(),
                                              gpuScene->MaxAccelIds(),
                                              gpuScene->BaseBoundaryMaterial());
    ERROR_CHECK(SceneError, scnE);

    // Visor Input Generation
    visorInput = std::make_unique<VisorWindowInput>(1.0, 1.0, 2.0,
                                                    gpuScene->CamerasCPU()[0]);

    // Window Params
    VisorOptions visorOpts;
    visorOpts.stereoOn = false;
    visorOpts.vSyncOn = false;
    visorOpts.eventBufferSize = 128;
    visorOpts.fpsLimit = 24.0f;

    visorOpts.wSize = Vector2i{1024, 1024};
    visorOpts.wFormat = IMAGE_PIXEL_FORMAT;


    visorOpts.iFormat = IMAGE_PIXEL_FORMAT;
    visorOpts.iSize = IMAGE_RESOLUTION;

    // Create Visor
    visorView = CreateVisorGL(visorOpts);
    visorView->SetInputScheme(*visorInput);

    // Attach the logic & Image format
    tracerBase = std::make_unique<TracerBase>();
    tracerBase->AttachLogic(*logic);
    tracerBase->SetImagePixelFormat(IMAGE_PIXEL_FORMAT);
    tracerBase->ResizeImage(IMAGE_RESOLUTION);
    tracerBase->ReportionImage();
    tracerBase->ResetImage();
    tracerBase->SetOptions(TRACER_OPTIONS);

    // Tracer Init
    TracerError trcE = tracerBase->Initialize(leaderDevice);
    ERROR_CHECK(TracerError, trcE);

    // Get a Self-Node
    // Generate your Node (in this case visor and renderer is on same node
    selfNode = std::make_unique<SelfNode>(*visorView, *tracerBase);
    visorInput->AttachVisorCallback(*selfNode);
    tracerBase->AttachTracerCallbacks(*selfNode);

    return true;
}

void SimpleTracerSetup::Body()
{
    GPUSceneI& scene = *gpuScene;

    CPUTimer t;
    t.Start();
    double elapsed = 0.0;
    
    // Specifically do not use self nodes loop functionality here
    // Main Poll Loop
    while(visorView->IsOpen())
    {
        // Run tracer
        tracerBase->GenerateInitialRays(scene, 0, 1);
        while(tracerBase->Continue())
        {
            tracerBase->Render();
        }
        tracerBase->FinishSamples();

        // Before try to show do render loop
        visorView->Render();

        // Present Back Buffer
        visorView->ProcessInputs();

        // Timing and Window Termination
        t.Lap();
        elapsed += t.Elapsed<CPUTimeSeconds>();
        if(elapsed >= WINDOW_DURATION) break;
    }
}