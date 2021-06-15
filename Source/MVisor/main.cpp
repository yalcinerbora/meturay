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

    bool guiderDebug;
    std::string guideDebugConfig;

    // Command Line Arguments
    CLI::App app{header};
    app.footer(ProgramConstants::Footer);

    CLI::Option* gdbConfOpt = app.add_option("-gdbc, --guideDebugConfig", guideDebugConfig, "Guider debugging configuration file");

    app.add_option("-gdb,--guideDebug", guiderDebug, "Debug visualize provided path guiders")
        ->needs(gdbConfOpt)
        ->expected(1)
        ->check(CLI::ExistingFile);

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


    return 0;
}