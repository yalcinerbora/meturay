#include "SelfNode.h"
#include "Camera.h"
#include "TracerStructs.h"
#include "TracerError.h"
#include "AnalyticData.h"
#include "Log.h"

#include "VisorI.h"
#include "TracerI.h"

SelfNode::SelfNode(VisorI& v, TracerI& t)
    : visor(v)
    , tracer(t)
{}

void SelfNode::SendScene(const std::string s)
{

}

void SelfNode::SendTime(const double)
{

}

void SelfNode::IncreaseTime(const double)
{

}

void SelfNode::DecreaseTime(const double)
{

}

void SelfNode::SendCamera(const CameraPerspective c)
{

}

void SelfNode::SendOptions(const TracerOptions opts)
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

}

bool SelfNode::Loop()
{ 
    return false; 
}