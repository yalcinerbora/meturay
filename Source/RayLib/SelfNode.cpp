#include "SelfNode.h"
#include "Camera.h"
#include "TracerStructs.h"
#include "TracerError.h"
#include "AnalyticData.h"
#include "Log.h"
#include "CPUTimer.h"

#include "VisorI.h"
#include "TracerI.h"

SelfNode::SelfNode(VisorI& v, TracerI& t)
    : visor(v)
    , tracer(t)
{}

void SelfNode::ChangeScene(const std::string s)
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

void SelfNode::ChangeCamera(const CameraPerspective c)
{

}

void SelfNode::ChangeCamera(const unsigned int)
{

}

void SelfNode::ChangeOptions(const TracerOptions opts)
{
    tracer.SetOptions(opts);
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
}

void SelfNode::SendLog(const std::string s)
{
    METU_LOG("Tracer: %s", s.c_str());
}

void SelfNode::SendError(TracerError err)
{
    METU_ERROR_LOG("Tracer Error: %s", static_cast<std::string>(err).c_str());
}

void SelfNode::SendAnalyticData(AnalyticData data)
{
    //TODO:

}

void SelfNode::SendImage(const std::vector<Byte> data,
                         PixelFormat f, int sampleCount,
                         Vector2i start, Vector2i end)
{
    visor.AccumulatePortion(std::move(data), f, sampleCount, start, end);
}

void SelfNode::SendAccelerator(HitKey key, const std::vector<Byte> data)
{
    // Do nothing since there is single tracer
    // No other tracer should be available to ask an acceleretor
}

void SelfNode::SendBaseAccelerator(const std::vector<Byte> data)
{
    // Do nothing since there is single tracer
    // No other tracer should be available to ask an acceleretor
}

// From Node Interface
NodeError SelfNode::Initialize()
{
    // OGL always wants to be in main thread

    // Create a tracer thread and make run along
    // Create an interface

    return NodeError::OK;
}

void SelfNode::Work()
{ 
    //GPUSceneI& scene = *gpuScene;

    // Specifically do not use self nodes loop functionality here
    // Main Poll Loop
    while(visor.IsOpen())
    {
        // Run tracer
        //tracer.GenerateInitialRays(scene, 0, 1);
        while(tracer.Continue())
        {
            tracer.Render();
        }
        tracer.FinishSamples();

        // Before try to show do render loop
        tracer.Render();

        // Present Back Buffer
        visor.ProcessInputs();
    }
}