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
#include "RayLib/GPUTracerI.h"

// Visor
#include "RayLib/VisorI.h"
#include "RayLib/VisorWindowInput.h"
#include "RayLib/MovementSchemes.h"
#include "VisorGL/VisorGLEntry.h"

// Tracer
#include "TracerLib/GPUSceneJson.h"
#include "TracerLib/ScenePartitioner.h"
#include "TracerLib/TracerLogicGenerator.h"

// Node
#include "RayLib/VisorCallbacksI.h"
#include "RayLib/TracerCallbacksI.h"
#include "RayLib/NodeI.h"
#include "RayLib/AnalyticData.h"
#include "RayLib/TracerError.h"
#include "RayLib/TracerOptions.h"

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
    public:
        static constexpr uint32_t       MAX_BOUNCES = 16;
        static constexpr int            SAMPLE_COUNT = 3;

        //static constexpr Vector2i       IMAGE_RESOLUTION = {16, 9};
        //static constexpr Vector2i       IMAGE_RESOLUTION = {32, 18};
        //static constexpr Vector2i       IMAGE_RESOLUTION = {320, 180};
        //static constexpr Vector2i       IMAGE_RESOLUTION = {640, 360};
        static constexpr Vector2i       IMAGE_RESOLUTION = {1280, 720};
        //static constexpr Vector2i       IMAGE_RESOLUTION = {1600, 900};
        //static constexpr Vector2i       IMAGE_RESOLUTION = {1920, 1080};
        //static constexpr Vector2i       IMAGE_RESOLUTION = {3840, 2160};

    private:
        const double                    Duration;

        VisorI& visor;
        GPUTracerI& tracer;
        GPUSceneI& scene;

    protected:
    public:
        // Constructor & Destructor
        MockNode(VisorI&, GPUTracerI&, GPUSceneI&,
                 double duration);
        ~MockNode() = default;

        // From Command Callbacks
        void        ChangeScene(const std::string) override {}
        void        ChangeTime(const double) override {}
        void        IncreaseTime(const double) override {}
        void        DecreaseTime(const double) override {}
        void        ChangeCamera(const CPUCamera) override {}
        void        ChangeCamera(const unsigned int) override {}
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
        void        SendCrashSignal() override {}
        void        SendLog(const std::string) override;
        void        SendError(TracerError) override;
        void        SendAnalyticData(AnalyticData) override {}
        void        SendImage(const std::vector<Byte> data,
                              PixelFormat, size_t offset,
                              Vector2i start = Zero2i,
                              Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void        SendCurrentOptions(TracerOptions) {};
        void        SendCurrentParameters(TracerParameters) {};

        // From Node Interface
        NodeError   Initialize() override { return NodeError::OK; }
        void        Work() override;
};

inline MockNode::MockNode(VisorI& v, GPUTracerI& t, 
                          GPUSceneI& s, double duration)
    : visor(v)
    , tracer(t)
    , scene(s)
    , Duration(duration)
{}

inline void MockNode::SendLog(const std::string s)
{
    METU_LOG("Tracer: %s", s.c_str());
}

inline void MockNode::SendError(TracerError err)
{
    METU_ERROR_LOG("Tracer: %s", static_cast<std::string>(err).c_str());
}

inline void MockNode::SendImage(const std::vector<Byte> data,
                                PixelFormat f, size_t offset,
                                Vector2i start, Vector2i end)
{
    visor.AccumulatePortion(std::move(data), f, offset, start, end);
}

inline void MockNode::Work()
{
    Utility::CPUTimer t;
    t.Start();
    double elapsed = 0.0;

    // Specifically do not use self nodes loop functionality here
    // Main Poll Loop
    while(visor.IsOpen())
    {
        // Run tracer
        tracer.GenerateWork(0);
        
        // Render
        while(tracer.Render());
        // Finalize (send image to visor)
        tracer.Finalize();
        //printf("\n----------------------------------\n");

        // Render scene window
        visor.Render();

        // Present Back Buffer
        visor.ProcessInputs();

        // Timing and Window Termination
        t.Lap();
        elapsed += t.Elapsed<CPUTimeSeconds>();
        
        double elapsedS = t.Elapsed<CPUTimeSeconds>();
        double rps = SAMPLE_COUNT * SAMPLE_COUNT * IMAGE_RESOLUTION[0] * IMAGE_RESOLUTION[1];
        rps *= (1.0 / elapsedS);
        rps /= 1'000'000.0;

        fprintf(stdout, "%c[2K", 27);
        fprintf(stdout, "Time: %fs Rps: %fM ray/s  Total: %fm\r", 
                elapsedS, rps, (elapsed / 60.0));
        //if(elapsed >= Duration) break;
    }
    METU_LOG("\n");
}

class SimpleTracerSetup
{
    private:        
        static constexpr Vector2i           SCREEN_RESOLUTION = {1280, 720};
       
        static constexpr double             WINDOW_DURATION = 3.5;
        static constexpr PixelFormat        IMAGE_PIXEL_FORMAT = PixelFormat::RGBA_FLOAT;

        static constexpr const char*        ESTIMATOR_TYPE = "BasicEstimator";
        static constexpr const char*        TRACER_TYPE = "BasicTracer";

        static constexpr const char*        TRACER_DLL = "Tracer-Test";
        static constexpr const char*        TRACER_LOGIC_POOL_GEN = "GenerateTestTracerPool";
        static constexpr const char*        TRACER_LOGIC_POOL_DEL = "DeleteTestTracerPool";
        static constexpr const char*        TRACER_MAT_POOL_GEN = "GenerateTestMaterialPool";
        static constexpr const char*        TRACER_MAT_POOL_DEL = "DeleteTestMaterialPool";

        static constexpr const char*        SURF_LOAD_DLL = "AssimpSurfaceLoaders";
        static constexpr const char*        SURF_LOAD_GEN = "GenerateAssimpPool";
        static constexpr const char*        SURF_LOAD_DEL = "DeleteAssimpPool";


        static constexpr TracerParameters   TRACER_PARAMETERS = 
        {
            false,  // Verbose
            0       // Seed
        };

        // Surface Loader Generators (Classes of primitive file loaders)
        SurfaceLoaderGenerator              surfaceLoaders;
        // Tracer Logic Generators (Classes of CUDA Coda Segments)
        TracerLogicGenerator                generator;

        // Scene Tracer and Visor
        GPUTracerPtr                        tracer;
        std::unique_ptr<VisorI>             visorView;
        std::unique_ptr<GPUSceneJson>       gpuScene;
        std::unique_ptr<CudaSystem>         cudaSystem;

        // Visor Input
        std::unique_ptr<VisorWindowInput>   visorInput;

        // Self Node
        std::unique_ptr<MockNode>           node;

        std::string                         tracerType;
        const std::u8string                 sceneName;
        const double                        sceneTime;

    public:
        // Constructors & Destructor
                            SimpleTracerSetup(std::string tracerType,
                                              std::u8string sceneName,
                                              double sceneTime);
                            SimpleTracerSetup() = default;

        bool                Init();
        void                Body();
};

inline SimpleTracerSetup::SimpleTracerSetup(std::string tracerType,
                                            std::u8string sceneName,
                                            double sceneTime)
    : sceneName(sceneName)
    , sceneTime(sceneTime)
    , tracerType(tracerType)
    , visorView(nullptr)
    , gpuScene(nullptr)
    , visorInput(nullptr)
    , node(nullptr)
    , tracer(nullptr, nullptr)
{}

inline bool SimpleTracerSetup::Init()
{
    TracerStatus status =
    {
        sceneName,
        0,

        0,

        CPUCamera{},

        MockNode::IMAGE_RESOLUTION,
        IMAGE_PIXEL_FORMAT,

        0.0,

        false,
        false,

        TRACER_PARAMETERS
    };   
    // Load Materials from Test-Material Shared Library
    DLLError dllE = generator.IncludeMaterialsFromDLL(TRACER_DLL, ".*",
                                                      SharedLibArgs{TRACER_MAT_POOL_GEN, TRACER_MAT_POOL_DEL});
    ERROR_CHECK(DLLError, dllE);
    // Load Tracer Logics from Test-Material Shared Library
    dllE = generator.IncludeTracersFromDLL(TRACER_DLL, ".*",
                                           SharedLibArgs{TRACER_LOGIC_POOL_GEN, TRACER_LOGIC_POOL_DEL});
    ERROR_CHECK(DLLError, dllE);
    // Load Assimp Surface Loader for loading files
    dllE = surfaceLoaders.IncludeLoadersFromDLL(SURF_LOAD_DLL, ".*",
                                                SharedLibArgs{SURF_LOAD_GEN, SURF_LOAD_DEL});
    ERROR_CHECK(DLLError, dllE);

    // Load GFG Surface Loader for gfg data
    // TODO:

    // Generate GPU List & A Partitioner
    // Check cuda system error here
    cudaSystem = std::make_unique<CudaSystem>();
    CudaError cudaE = cudaSystem->Initialize();
    ERROR_CHECK(CudaError, cudaE);

    // GPU Data Partitioner
    // Basically delegates all work to single GPU
    SingleGPUScenePartitioner partitioner(*cudaSystem);

    // Load Scene & get material/primitive mappings
    gpuScene = std::make_unique<GPUSceneJson>(sceneName,
                                              partitioner,
                                              generator,
                                              surfaceLoaders,
                                              *cudaSystem);
    SceneError scnE = gpuScene->LoadScene(sceneTime);
    ERROR_CHECK(SceneError, scnE);

    MovementScemeList MovementSchemeList = {};    
    KeyboardKeyBindings KeyBinds = VisorConstants::DefaultKeyBinds;
    MouseKeyBindings MouseBinds = VisorConstants::DefaultButtonBinds;

    // Visor Input Generation
    visorInput = std::make_unique<VisorWindowInput>(std::move(KeyBinds),
                                                    std::move(MouseBinds),
                                                    std::move(MovementSchemeList),
                                                    CPUCamera{});
                                                    
    // Window Params
    VisorOptions visorOpts;
    visorOpts.stereoOn = false;
    visorOpts.vSyncOn = false;
    visorOpts.eventBufferSize = 128;
    visorOpts.fpsLimit = 24.0f;

    visorOpts.wSize = SCREEN_RESOLUTION;
    visorOpts.wFormat = IMAGE_PIXEL_FORMAT;


    visorOpts.iFormat = IMAGE_PIXEL_FORMAT;
    visorOpts.iSize = MockNode::IMAGE_RESOLUTION;

    // Create Visor
    visorView = CreateVisorGL(visorOpts);
    visorView->SetInputScheme(*visorInput);

    // Set Window Res wrt to monitor resolution
    Vector2i newImgSize = 3 * visorView->MonitorResolution() / 5;
    visorView->SetWindowSize(newImgSize);

    // Generate Tracer Object
    scnE = generator.GenerateTracer(tracer,
                                    *cudaSystem,
                                    *gpuScene,
                                    TRACER_PARAMETERS,
                                    tracerType);
    ERROR_CHECK(SceneError, scnE);
    tracer->SetImagePixelFormat(IMAGE_PIXEL_FORMAT);
    tracer->ResizeImage(MockNode::IMAGE_RESOLUTION);
    tracer->ReportionImage();
    tracer->ResetImage();
    
    // Set Options
    const TracerOptions opts = TracerOptions(
    {
        {"Samples", OptionVariable(MockNode::SAMPLE_COUNT)},
        {"MaxDepth", OptionVariable(MockNode::MAX_BOUNCES)},
        {"NextEventEstimation", OptionVariable(false)},
        {"RussianRouletteStart", OptionVariable(3u)}
    });
    TracerError trcE = tracer->SetOptions(opts);
    ERROR_CHECK(TracerError, trcE);

    // Tracer Init
    trcE = tracer->Initialize();
    ERROR_CHECK(TracerError, trcE);

    // Get a Self-Node
    // Generate your Node (in this case visor and renderer is on same node
    node = std::make_unique<MockNode>(*visorView, *tracer, *gpuScene,
                                      WINDOW_DURATION);
    NodeError nodeE = node->Initialize();
    ERROR_CHECK(NodeError, nodeE);
    visorInput->AttachVisorCallback(*node);
    tracer->AttachTracerCallbacks(*node);

    return true;
}

inline void SimpleTracerSetup::Body()
{
    try
    {
        node->Work();
    }
    catch(TracerException const& e)
    {
        std::string err = static_cast<TracerError>(e);
        METU_ERROR_LOG("%s (%s)", err.c_str(), e.what());
        GTEST_FAIL();
    }
}