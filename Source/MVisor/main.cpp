#include <array>
#include <iostream>

#include "RayLib/System.h"
#include "RayLib/Log.h"

// Visor
#include "RayLib/VisorI.h"
#include "RayLib/VisorWindowInput.h"
#include "RayLib/MovementSchemes.h"

// Args Parser
#include <CLI11.hpp>

#include <array>

int main(int argc, const char* argv[])
{
    // Fancy CMD
    EnableVTMode();

    std::string visorConfigFileName;

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

    app.add_option("--gdbc,--guideDebugConfig",
                   guideDebugConfig,
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

        // Initialize a Visor











    }
    else
    {
        METU_ERROR_LOG("MVisor currently only support path guider visualzation..");
    }

    return 0;
}