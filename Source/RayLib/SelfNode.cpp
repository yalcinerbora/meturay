#include "SelfNode.h"

#include "TracerOptions.h"
#include "VisorCamera.h"
#include "AnalyticData.h"
#include "VisorI.h"

SelfNode::SelfNode(VisorI& v, TracerSystemI& t,
                   const TracerOptions& opts,
                   const TracerParameters& params,
                   const std::string& tracerTypeName,
                   const Vector2i& resolution)
    : tracerThread(t, opts, params, *this, tracerTypeName)
    , visor(v)
{
    tracerThread.SetImageResolution(resolution);
    // Self Node has only one tracer
    // so set image portion to resolution
    tracerThread.SetImagePortion();
}

void SelfNode::ChangeScene(const std::u8string s)
{
    tracerThread.SetScene(s);
}

void SelfNode::ChangeTime(const double t)
{
    tracerThread.ChangeTime(t);
}

void SelfNode::IncreaseTime(const double t)
{
    tracerThread.IncreaseTime(t);
}

void SelfNode::DecreaseTime(const double t)
{
    tracerThread.DecreaseTime(t);
}

void SelfNode::ChangeCamera(const VisorCamera c)
{
    tracerThread.ChangeCamera(c);
}

void SelfNode::ChangeCamera(const unsigned int cameraId)
{
    tracerThread.ChangeCamera(cameraId);
}

void SelfNode::StartStopTrace(const bool)
{
    // TOD:: Adjust tracer thread
}

void SelfNode::PauseContTrace(const bool)
{
    //TODO: Adjust tracer thread
}


void SelfNode::WindowMinimizeAction(bool minimized)
{
    // TODO:
}

void SelfNode::WindowCloseAction()
{
    // TODO:: Terminate the tracer thread
    tracerThread.Stop();
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
    visor.AccumulatePortion(std::move(data), f, offset, start, end);
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
    // Start tracer thread
    tracerThread.Start();

    // Set Rendering context on main thread
    visor.SetRenderingContextCurrent();

    return NodeError::OK;
}

void SelfNode::Work()
{ 
    while(visor.IsOpen() &&
          !tracerThread.IsTerminated())
    {
        // Render Loop
        visor.Render();

        // Process Inputs MUST be called on main thread
        // since Windows OS event poll is required to be called
        // on main thread, I don't know about other operating systems
        //
        // OGL Visor will use GLFW for window operations
        // and it also requires "glfwPollEvents()" function
        // (which this function calls it internally)
        // to be called on main thread
        visor.ProcessInputs();
    }
        
    // Visor is closed terminate tracer thread
    tracerThread.Stop();    
}