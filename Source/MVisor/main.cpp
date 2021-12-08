#include <array>
#include <iostream>

#include "RayLib/System.h"
#include "RayLib/Log.h"
#include "RayLib/SharedLib.h"

// Visor
#include "RayLib/VisorI.h"
#include "RayLib/VisorCallbacksI.h"
#include "RayLib/MovementSchemes.h"
#include "RayLib/VisorError.h"

// Args Parser
#include <CLI11.hpp>

#include <array>

int main(int argc, const char* argv[])
{
    // Fancy CMD
    EnableVTMode();

    // Error structs
    DLLError dError = DLLError::OK;

    // Arg Parsing
    // Header
    const std::string BundleName = ProgramConstants::ProgramName;
    const std::string AppName = "MVisor";
    const std::string Description = "Tracer or Debug Visualizer";
    const std::string header = (BundleName + " - " + AppName + " " + Description);

    bool guiderDebug = false;
    std::string guideDebugConfig = "";

    // Command Line Arguments
    CLI::App app{header};
    app.footer(ProgramConstants::Footer);

    CLI::Option* debugOptOn = app.add_flag("--gdb,--guideDebug", guiderDebug,
                                             "Visualize path guiders provided in the config file");
    app.add_option("--gdbc,--guideDebugConfig", guideDebugConfig,
                   "Guider debugging configuration file")
        ->check(CLI::ExistingFile)
        ->needs(debugOptOn);

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

    // Load VisorGL
    if(guiderDebug)
    {
        // Load VisorGL.dll
        SharedLib visorDLL("VisorGL");
        // Initialize a Visor
        SharedLibPtr<VisorI> debugVisor = {nullptr, nullptr};
        dError = visorDLL.GenerateObjectWithArgs(debugVisor,
                                                 SharedLibArgs{
                                                    "CreateGuideDebugGL",
                                                    "DeleteVisorGL"},
                                                 // Args
                                                 Vector2i(1600, 900),
                                                 guideDebugConfig);
        ERROR_CHECK_INT(DLLError, dError);

        // Set an input scheme
        EmptyVisorCallback emptyCallback;
        KeyboardKeyBindings kb = VisorConstants::DefaultKeyBinds;
        MouseKeyBindings bb = VisorConstants::DefaultButtonBinds;

        // Init Visor with input scheme
        VisorError vError = debugVisor->Initialize(emptyCallback, kb, bb, {});
        ERROR_CHECK_INT(VisorError, vError);

        // Render Loop
        while(debugVisor->IsOpen())
        {
            debugVisor->Render();
            debugVisor->ProcessInputs();
        }

        // Orderly Delete Unique Ptrs
        debugVisor = SharedLibPtr<VisorI>(nullptr, nullptr);
        return 0;
    }
    else
    {
        METU_ERROR_LOG("MVisor currently only support path guider visualization..");
    }

    return 0;
}