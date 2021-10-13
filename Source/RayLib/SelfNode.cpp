#include "SelfNode.h"

#include "TracerOptions.h"
#include "VisorTransform.h"
#include "AnalyticData.h"
#include "VisorI.h"

SelfNode::SelfNode(VisorI& v, TracerSystemI& t,
                   const TracerOptions& opts,
                   const TracerParameters& params,
                   const std::string& tracerTypeName,
                   const Vector2i& resolution)
    : tracerThread(t, opts, params, *this, tracerTypeName)
    , visor(v)
    , tracerCrashed(false)
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

void SelfNode::ChangeCamera(const VisorTransform t)
{
    //std::cout << "Gaze: ["
    //          << c.gazePoint[0] << ", "
    //          << c.gazePoint[1] << ", "
    //          << c.gazePoint[2] << "]" << std::endl;
    //std::cout << "Pos: ["
    //          << c.position[0] << ", "
    //          << c.position[1] << ", "
    //          << c.position[2] << "]" << std::endl;
    //std::cout << "Up: ["
    //          << c.up[0] << ", "
    //          << c.up[1] << ", "
    //          << c.up[2] << "]" << std::endl;
    tracerThread.ChangeTransform(t);
}

void SelfNode::ChangeCamera(const unsigned int cameraId)
{
    tracerThread.ChangeCamera(cameraId);
}

void SelfNode::StartStopTrace(const bool started)
{
    SendLog(std::string("Tracer is ") + ((started) ? "started" : "stopped"));
    tracerThread.StartStopTrace(started);
}

void SelfNode::PauseContTrace(const bool paused)
{
    SendLog(std::string("Tracer is ") + ((paused) ? "paused" : "continued"));
    tracerThread.PauseContTrace(paused);
}

void SelfNode::WindowMinimizeAction(bool minimized)
{
    // TODO:
}

void SelfNode::WindowCloseAction()
{
    // Set a variable

    // TODO:: Terminate the tracer thread
    //tracerThread.Stop();
}

void SelfNode::SendCrashSignal()
{
    tracerCrashed = true;
}

void SelfNode::SendLog(const std::string s)
{
    METU_LOG("Tracer: {:s}", s);
}

void SelfNode::SendError(TracerError err)
{
    METU_ERROR_LOG("Tracer: {:s}", static_cast<std::string>(err));
}

void SelfNode::SendAnalyticData(AnalyticData data)
{
    //TODO:
}

void SelfNode::SendImageSectionReset(Vector2i start, Vector2i end)
{
    visor.ResetSamples(start, end);
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

void SelfNode::SendCurrentTransform(VisorTransform t)
{
    visor.SetTransform(t);
}

void SelfNode::SendCurrentSceneCameraCount(uint32_t camCount)
{
    visor.SetSceneCameraCount(camCount);
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
    // Self Node will terminate if
    while(// Visor is Closed
          visor.IsOpen() &&
          // Tracer Thread unable to do stuff
          // (i.e. unable to load a tracer scene etc)
          !tracerThread.IsTerminated() &&
          // Tracer itself is crashed (some internal runtime error etc)
          !tracerCrashed)
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