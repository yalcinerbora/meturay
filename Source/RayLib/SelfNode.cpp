#include "SelfNode.h"
#include "Camera.h"
#include "TracerStructs.h"
#include "TracerError.h"
#include "AnalyticData.h"

//#include "VisorI.h"

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

void SelfNode::SendCamera(const CameraPerspective)
{

}

void SelfNode::SendOptions(const TracerOptions)
{

}

void SelfNode::StartStopTrace(const bool)
{

}

void SelfNode::PauseContTrace(const bool)
{

}

void SelfNode::WindowMinimizeAction(bool minimized)
{

}

void SelfNode::WindowCloseAction()
{

}

void SelfNode::SendError(TracerError err)
{

}

void SelfNode::SendAnalyticData(AnalyticData data)
{

}

void SelfNode::SendImage(const std::vector<Byte> data,
					  PixelFormat, int sampleCount,
					  Vector2i start, Vector2i end)
{

}

void SelfNode::SendAccelerator(HitKey key, const std::vector<Byte> data)
{

}

void SelfNode::SendBaseAccelerator(const std::vector<Byte> data)
{

}