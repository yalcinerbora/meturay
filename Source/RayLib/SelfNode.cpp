#include "SelfNode.h"

#include "TracerOptions.h"
#include "VisorCamera.h"
#include "AnalyticData.h"


//#include "TracerStructs.h"
//#include "TracerError.h"



#include "Log.h"
//#include "CPUTimer.h"

//#include "VisorI.h"
//#include "GPUTracerI.h"



SelfNode::SelfNode(VisorI& v, GPUTracerI& t)
    : visorThread(v)
    , tracerThread(t)
{}

void SelfNode::ChangeScene(const std::u8string s)
{

}

void SelfNode::ChangeTime(const double)
{

}

void SelfNode::IncreaseTime(const double)
{

}

void SelfNode::DecreaseTime(const double)
{

}

void SelfNode::ChangeCamera(const VisorCamera c)
{

}

void SelfNode::ChangeCamera(const unsigned int cameraId)
{

}

void SelfNode::StartStopTrace(const bool)
{
    // TOD:: Adjust tracer thread
}

void SelfNode::PauseContTrace(const bool)
{
    //TODO: Adjust tracer thread
}

void SelfNode::SetTimeIncrement(const double)
{

}

void SelfNode::SaveImage()
{

}

void SelfNode::SaveImage(const std::string& path,
                         const std::string& fileName,
                         ImageType,
                         bool overwriteFile)
{

}

void SelfNode::WindowMinimizeAction(bool minimized)
{
    // TODO:
}

void SelfNode::WindowCloseAction()
{
    // TODO:: Terminate the tracer thread
}

void SelfNode::SendCrashSignal()
{

}

void SelfNode::SendLog(const std::string s)
{
    METU_LOG("Tracer: %s", s.c_str());
}

void SelfNode::SendError(TracerError err)
{
    METU_ERROR_LOG("Tracer: %s", static_cast<std::string>(err).c_str());
}

void SelfNode::SendAnalyticData(AnalyticData data)
{
    //TODO:

}

void SelfNode::SendImage(const std::vector<Byte> data,
                         PixelFormat f, size_t offset,
                         Vector2i start, Vector2i end)
{
    visorThread.AccumulateImagePortion(std::move(data), f, offset, start, end);
}

void SelfNode::SendCurrentOptions(TracerOptions)
{
    // We have only one trader so no delegation to other tracers
    // Only refresh the visor if it shows the current options

    //visor.
}

void SelfNode::SendCurrentParameters(TracerParameters)
{
    // Same as Tracer Options
}

// From Node Interface
NodeError SelfNode::Initialize()
{

    // Create a tracer thread and make run along
    // Create an interface


    tracerThread.Start();
    visorThread.Start();
    return NodeError::OK;
}

void SelfNode::Work()
{ 
    while(!visorThread.IsTerminated() &&
          !tracerThread.IsTerminated())
    {
        // Process Inputs MUST be called on main thread
        // since Windows OS event poll is required to be called
        // on main thread, I don't know about other operating systems
        //
        // OGL Visor will use GLFW for window operations
        // and it also requires "glfwPollEvents()" function
        // (which this function calls it internally)
        // to be called on main thread
        visorThread.ProcessInputs();
    }
        
    // Visor thread is closed terminate tracer thread
    tracerThread.Stop();

    // All done!

    ////GPUSceneI& scene = *gpuScene;

    //// Specifically do not use self nodes loop functionality here
    //// Main Poll Loop
    //while(visor.IsOpen())
    //{
    //    // Run tracer
    //    //tracer.GenerateInitialRays(scene, 0, 1);
    //    while(tracer.Render())
    //    {
    //        // Node should check if it is requested to do stuff
    //    }
    //    tracer.Finalize();

    //    // Before try to show do render loop
    //    tracer.Render();

    //    // Present Back Buffer
    //    visor.ProcessInputs();
    //}
}