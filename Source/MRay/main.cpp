// Common
#include "RayLib/System.h"
#include "RayLib/Log.h"
#include "RayLib/UTF8StringConversion.h"
#include "RayLib/DLLError.h"
#include "RayLib/SharedLib.h"
#include "RayLib/ConfigParsers.h"
// Visor
#include "RayLib/VisorI.h"
#include "RayLib/MovementSchemes.h"
#include "RayLib/VisorError.h"
// Tracer
#include "RayLib/Options.h"
#include "RayLib/TracerSystemI.h"
// Node
#include "RayLib/SelfNode.h"
// Args Parser
#include <CLI/CLI.hpp>
#include <array>

#include <random>

#include "RayLib/HemiDistribution.h"

//Vector3 lambert(float& pdf,
//                Vector3& wo,
//                const Vector3f& wi,
//                const Vector2f& xi)
//{
//    // Ray Selection
//    Vector3 normal = ZAxis;
//    // Generate New Ray Direction (This is in tangent space)
//    wo = HemiDistribution::HemiCosineCDF(xi, pdf);
//    wo.NormalizeSelf();
//    // Cos Theta
//    float nDotL = std::max(normal.Dot(wo), 0.0f);
//    // Radiance Calculation
//    const Vector3f albedo = Vector3f(1.0f);
//    return nDotL * albedo * MathConstants::InvPi;
//}

int main(int argc, const char* argv[])
{
    // Fancy CMD
    EnableVTMode();


    //std::mt19937 rng;
    //std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    //Vector3f val = Zero3f;
    //Vector3f dir = Zero3f;
    //for(int i = 0; i < 10'000; i++)
    //{
    //    float pdf;
    //    Vector2f xi(dist(rng), dist(rng));
    //    Vector3f wo;
    //    Vector3f out = lambert(pdf, wo, Zero3f, xi);
    //    float iF = static_cast<float>(i);

    //    out /= pdf;

    //    dir = (dir * iF + wo) / (iF + 1);
    //    val = (val * iF + out) / (iF + 1);
    //    METU_LOG("dir({}, {}, {}), val({},{},{})",
    //             dir[0], dir[1], dir[2],
    //             val[0], val[1], val[2]);
    //}


    std::array<int, 2> resolution;
    std::string tracerConfigFileName;
    std::string visorConfigFileName;
    std::string sceneFileName = "";

    // Header
    const std::string BundleName = ProgramConstants::ProgramName;
    const std::string AppName = "MRay";
    const std::string Description = "Single Platform GPU Renderer and Visualizer";
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
        METU_LOG(app.help());
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
    VisorError vError = VisorError::OK;

    // Variables Related to the MRay
    SharedLibPtr<TracerSystemI> tracerSystem = {nullptr, nullptr};
    SharedLibPtr<GPUTracerI> tracer = {nullptr, nullptr};

    // Parse Tracer Config
    Options filterOptions;
    Options tracerOptions;
    TracerParameters tracerParameters;
    SharedLibArgs tracerDLLEntryFunctionNames;
    std::string tracerDLLName;
    std::string tracerTypeName;
    std::vector<SurfaceLoaderSharedLib> surfaceLoaderLibraries;
    ScenePartitionerType scenePartitionType;

    if(!ConfigParser::ParseTracerOptions(tracerOptions, tracerParameters,
                                         tracerTypeName, tracerDLLName,
                                         tracerDLLEntryFunctionNames,
                                         filterOptions,
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
    MovementSchemeList movementSchemeList;
    KeyboardKeyBindings keyBinds;
    MouseKeyBindings mouseBinds;

    if(!ConfigParser::ParseVisorOptions(keyBinds, mouseBinds,
                                        movementSchemeList,
                                        visorOpts, visorDLLName,
                                        visorDLLEntryFunctionNames,
                                        //
                                        visorConfigFileName))
    {
        METU_ERROR_LOG("Unable to parse Visor Options");
        return 1;
    }

    // Generate Visor
    // Attach visors window callbacks to this input scheme
    // Additionally attach this specific visor to this?????????????
    SharedLib visorDLL(visorDLLName);
    SharedLibPtr<VisorI> visor = {nullptr, nullptr};
    dError = visorDLL.GenerateObjectWithArgs(visor, visorDLLEntryFunctionNames,
                                             // Args
                                             visorOpts,
                                             resolution,
                                             PixelFormat::RGBA_FLOAT);
    ERROR_CHECK_INT(DLLError, dError);

    // Generate Tracer
    SharedLib tracerDLL(tracerDLLName);
    dError = tracerDLL.GenerateObject(tracerSystem, tracerDLLEntryFunctionNames);
    ERROR_CHECK_INT(DLLError, dError);
    tError = tracerSystem->Initialize(surfaceLoaderLibraries, scenePartitionType);
    ERROR_CHECK_INT(TracerError, tError);

    // Create a Self Node
    SelfNode selfNode(*visor, *tracerSystem,
                      tracerOptions, tracerParameters,
                      tracerTypeName, filterOptions,
                      Vector2i(resolution.data()));

    // Init Visor with the self node (callbacks)
    vError = visor->Initialize(selfNode, keyBinds, mouseBinds, std::move(movementSchemeList));
    ERROR_CHECK_INT(VisorError, vError);

    // Init Self Node
    nError = selfNode.Initialize();
    ERROR_CHECK_INT(NodeError, nError);

    // Do work loop of the self node
    try
    {
        // If scene file is provided as a argument set scene for the node
        if(!sceneFileName.empty())
        {
            selfNode.ChangeScene(Utility::CopyStringU8(sceneFileName));
            selfNode.ChangeTime(0.0);
        }

        // Work returns when a crash occurs or user terminates
        selfNode.Work();
    }
    catch(TracerException const& e)
    {
        std::string err = static_cast<TracerError>(e);
        METU_ERROR_LOG("{:s} ({:s})", err.c_str(), e.what());
        return 1;
    }

    // Orderly Delete Unique Ptrs
    // Shared lib dependent ptrs should be released first
    tracerSystem = SharedLibPtr<TracerSystemI>(nullptr, nullptr);
    visor = SharedLibPtr<VisorI>(nullptr, nullptr);
    // Shared Libs etc. will destroy on scope exit
    return 0;
}