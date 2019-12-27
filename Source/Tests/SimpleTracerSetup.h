#pragma once

#include <gtest/gtest.h>

// Common
#include "RayLib/System.h"
#include "RayLib/Log.h"
#include "RayLib/Constants.h"
#include "RayLib/CPUTimer.h"
#include "RayLib/SurfaceLoaderGenerator.h"
#include "RayLib/TracerStatus.h"
#include "RayLib/DLLError.h"

// Visor
#include "RayLib/VisorI.h"
#include "RayLib/VisorWindowInput.h"
#include "RayLib/MovementSchemes.h"
#include "VisorGL/VisorGLEntry.h"

// Tracer
#include "TracerLib/GPUSceneJson.h"
#include "TracerLib/TracerLoader.h"
#include "TracerLib/TracerBase.h"
#include "TracerLib/ScenePartitioner.h"
#include "TracerLib/TracerLogicGenerator.h"

// Node
#include "RayLib/VisorCallbacksI.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/NodeI.h"
#include "RayLib/AnalyticData.h"
#include "RayLib/TracerError.h"

#define ERROR_CHECK(ErrType, e) \
if(e != ErrType::OK) \
{\
    METU_ERROR_LOG("%s", static_cast<std::string>(e).c_str()); \
    return false;\
}
class MockNode
    : public VisorCallbacksI
    , public TracerCallbacksI
    , public NodeI
{
    private:
        const double        Duration;

        VisorI&             visor;
        TracerI&            tracer;
        GPUSceneI&          scene;

    protected:
    public:
        // Constructor & Destructor
                    MockNode(VisorI&, TracerI&, GPUSceneI&,
                             double duration);
                    ~MockNode() = default;

        // From Command Callbacks
        void        ChangeScene(const std::string) override {}
        void        ChangeTime(const double) override {}
        void        IncreaseTime(const double) override {}
        void        DecreaseTime(const double) override {}
        void        ChangeCamera(const CameraPerspective) override {}
        void        ChangeCamera(const unsigned int) override {}
        void        ChangeOptions(const TracerOptions) override {}
        void        StartStopTrace(const bool) override {}
        void        PauseContTrace(const bool) override {}
        void        SetTimeIncrement(const double) override {}
        void        SaveImage() override {}
        void        SaveImage(const std::string& path,
                              const std::string& fileName,
                              ImageType,
                              bool overwriteFile) override {}

        void        WindowMinimizeAction(bool minimized) override {}
        void        WindowCloseAction() override {}

        // From Tracer Callbacks
        void        SendLog(const std::string) override;
        void        SendError(TracerError) override;
        void        SendAnalyticData(AnalyticData) override {}
        void        SendImage(const std::vector<Byte> data,
                              PixelFormat, int sampleCount,
                              Vector2i start = Zero2i,
                              Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void        SendAccelerator(HitKey key, const std::vector<Byte> data) override {}
        void        SendBaseAccelerator(const std::vector<Byte> data) override {}

        // From Node Interface
        NodeError   Initialize() override { return NodeError::OK; }
        void        Work() override;
};

MockNode::MockNode(VisorI& v, TracerI& t, GPUSceneI& s,
                   double duration)
    : visor(v)
    , tracer(t)
    , scene(s)
    , Duration(duration)
{}

void MockNode::SendLog(const std::string s)
{
    METU_LOG("Tracer: %s", s.c_str());
}

void MockNode::SendError(TracerError err)
{
    METU_ERROR_LOG("Tracer Error: %s", static_cast<std::string>(err).c_str());
}

void MockNode::SendImage(const std::vector<Byte> data,
                         PixelFormat f, int sampleCount,
                         Vector2i start, Vector2i end)
{
    visor.AccumulatePortion(std::move(data), f, sampleCount, start, end);
}

void MockNode::Work()
{
    Utility::CPUTimer t;
    t.Start();
    double elapsed = 0.0;

    // Specifically do not use self nodes loop functionality here
    // Main Poll Loop
    while(visor.IsOpen())
    {
        // Run tracer
        tracer.GenerateInitialRays(scene, 0, 1);
        
        uint32_t i = 0;
        while(tracer.Continue() && i < 4)
        {
            tracer.Render();
            i++;
        }
        tracer.FinishSamples();
        //printf("\n----------------------------------\n");

        // Before try to show do render loop
        visor.Render();

        // Present Back Buffer
        visor.ProcessInputs();

        // Timing and Window Termination
        t.Lap();
        elapsed += t.Elapsed<CPUTimeSeconds>();
        fprintf(stdout, "Time: %fs\r", t.Elapsed<CPUTimeSeconds>());        
        //if(elapsed >= Duration) break;
    }
    METU_LOG("\n");
}

class SimpleTracerSetup
{

    private:
        static constexpr Vector2i           SCREEN_RESOLUTION = {1024, 1024};
        static constexpr Vector2i           IMAGE_RESOLUTION = {256, 256};
        static constexpr double             WINDOW_DURATION = 3.5;
        static constexpr PixelFormat        IMAGE_PIXEL_FORMAT = PixelFormat::RGBA_FLOAT;

        static constexpr const char*        TRACER_DLL = "Tracer-Test";
        static constexpr const char*        TRACER_LOGIC_GEN = "GenerateBasicTracer";
        static constexpr const char*        TRACER_LOGIC_DEL = "DeleteBasicTracer";
        static constexpr const char*        TRACER_MAT_POOL_GEN = "GenerateTestMaterialPool";
        static constexpr const char*        TRACER_MAT_POOL_DEL = "DeleteTestMaterialPool";

        static constexpr const char*        SURF_LOAD_DLL = "AssimpSurfaceLoaders";
        static constexpr const char*        SURF_LOAD_GEN = "GenerateAssimpPool";
        static constexpr const char*        SURF_LOAD_DEL = "DeleteAssimpPool";


        static constexpr TracerParameters   TRACER_PARAMETERS = 
        {
            0   // Seed
        };
        static constexpr TracerOptions      TRACER_OPTIONS =
        {
            false   // Verbose
        };

        // Surface Loader Generators (Classes of primitive file loaders)
        SurfaceLoaderGenerator              surfaceLoaders;
        // Tracer Logic Generators (Classes of CUDA Coda Segments)
        TracerLogicGenerator                generator;

        // Scene Tracer and Visor
        std::unique_ptr<VisorI>             visorView;
        std::unique_ptr<TracerBase>         tracerBase;
        std::unique_ptr<GPUSceneJson>       gpuScene;
        std::unique_ptr<CudaSystem>         cudaSystem;

        // Visor Input
        std::unique_ptr<VisorWindowInput>   visorInput;

        // Self Node
        std::unique_ptr<MockNode>           node;

        const std::u8string                 sceneName;
        const double                        sceneTime;

    public:
        // Constructors & Destructor
                            SimpleTracerSetup(std::u8string sceneName,
                                              double sceneTime);
                            SimpleTracerSetup() = default;

        bool                Init();
        void                Body();
};

SimpleTracerSetup::SimpleTracerSetup(std::u8string sceneName, double sceneTime)
    : sceneName(sceneName)
    , sceneTime(sceneTime)
    , visorView(nullptr)
    , tracerBase(nullptr)
    , gpuScene(nullptr)
    , visorInput(nullptr)
    , node(nullptr)
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
    //sharedLib = std::make_unique<SharedLib>(TRACER_DLL);
    //tracerGenerator = TracerLoader::LoadTracerLogic(*sharedLib,
    //                                                TRACER_DLL_GENERATOR,
    //                                                TRACER_DLL_DELETER);
   
    // Load Materials from Test-Material Pool Shared Library
    DLLError dllE = generator.IncludeMaterialsFromDLL(TRACER_DLL, ".*",
                                                      SharedLibArgs{TRACER_MAT_POOL_GEN, TRACER_MAT_POOL_DEL});
    ERROR_CHECK(DLLError, dllE);

    // Load Obj Surface Loader (Required Only for Cornell Example)
    dllE = surfaceLoaders.IncludeLoadersFromDLL(SURF_LOAD_DLL, ".*",
                                                SharedLibArgs{SURF_LOAD_GEN, SURF_LOAD_DEL});
    ERROR_CHECK(DLLError, dllE);

    // Generate GPU List & A Partitioner
    // Check cuda system error here
    cudaSystem = std::make_unique<CudaSystem>();
    CudaError cudaE = cudaSystem->Initialize();
    ERROR_CHECK(CudaError, cudaE);
    //const CudaGPU& leaderDevice = *cudaSystem->GPUList().begin();

    // GPU Data Partitioner
    SingleGPUScenePartitioner partitioner(*cudaSystem);

    // Load Scene
    gpuScene = std::make_unique<GPUSceneJson>(sceneName,
                                              partitioner,
                                              generator,
                                              surfaceLoaders);
    SceneError scnE = gpuScene->LoadScene(sceneTime);
    ERROR_CHECK(SceneError, scnE);


    // Finally generate logic after successfull load
    TracerBaseLogicI* logic;
    dllE = generator.GenerateBaseLogic(logic, TRACER_PARAMETERS,
                                       gpuScene->MaxMatIds(),
                                       gpuScene->MaxAccelIds(),
                                       gpuScene->BaseBoundaryMaterial(),
                                       TRACER_DLL,
                                       SharedLibArgs{TRACER_LOGIC_GEN, TRACER_LOGIC_DEL});
    ERROR_CHECK(DLLError, dllE);

    MovementScemeList MovementSchemeList = {};    
    KeyboardKeyBindings KeyBinds = VisorConstants::DefaultKeyBinds;
    MouseKeyBindings MouseBinds = VisorConstants::DefaultButtonBinds;

    // Visor Input Generation
    visorInput = std::make_unique<VisorWindowInput>(std::move(KeyBinds),
                                                    std::move(MouseBinds),
                                                    std::move(MovementSchemeList),
                                                    gpuScene->CamerasCPU()[0]);
                                                    
    // Window Params
    VisorOptions visorOpts;
    visorOpts.stereoOn = false;
    visorOpts.vSyncOn = false;
    visorOpts.eventBufferSize = 128;
    visorOpts.fpsLimit = 24.0f;

    visorOpts.wSize = SCREEN_RESOLUTION;
    visorOpts.wFormat = IMAGE_PIXEL_FORMAT;


    visorOpts.iFormat = IMAGE_PIXEL_FORMAT;
    visorOpts.iSize = IMAGE_RESOLUTION;

    // Create Visor
    visorView = CreateVisorGL(visorOpts);
    visorView->SetInputScheme(*visorInput);

    // Attach the logic & Image format
    tracerBase = std::make_unique<TracerBase>(*cudaSystem);
    tracerBase->AttachLogic(*logic);
    tracerBase->SetImagePixelFormat(IMAGE_PIXEL_FORMAT);
    tracerBase->ResizeImage(IMAGE_RESOLUTION);
    tracerBase->ReportionImage();
    tracerBase->ResetImage();
    tracerBase->SetOptions(TRACER_OPTIONS);

    // Tracer Init
    TracerError trcE = tracerBase->Initialize();
    ERROR_CHECK(TracerError, trcE);

    // Get a Self-Node
    // Generate your Node (in this case visor and renderer is on same node
    node = std::make_unique<MockNode>(*visorView, *tracerBase, *gpuScene,
                                      WINDOW_DURATION);
    NodeError nodeE = node->Initialize();
    ERROR_CHECK(NodeError, nodeE);
    visorInput->AttachVisorCallback(*node);
    tracerBase->AttachTracerCallbacks(*node);

    return true;
}

void SimpleTracerSetup::Body()
{
    node->Work();
}