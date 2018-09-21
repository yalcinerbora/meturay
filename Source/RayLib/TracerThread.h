#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>

#include "LoopingThreadI.h"
#include "ThreadData.h"
#include "Camera.h"
#include "TracerStructs.h"

class SceneI;
class TracerI;
class TracerDistributorI;

/** Thread wrapper
	TracerI encapsulates system
	TracerThread is used to command Tracers using callbacks futures etc.
*/
class TracerThread : public LoopingThreadI
{
	private:
		struct ImageSegment
		{
			Vector2ui pixelStart;
			Vector2ui pixelCount;
		};

	private:
		// Actual Tracer
		TracerI&						tracer;

		// Current Settings
		ThreadData<double>				time;
		ThreadData<CameraPerspective>	camera;
		ThreadData<std::string>			scene;
		ThreadData<Vector2ui>			resolution;
		ThreadData<uint32_t>			sample;
		ThreadData<TracerParameters>	parameters;
		ThreadData<ImageSegment>		segment;
		
		int								currentFPS;
		int								currentFrame;

		// Thread work
		void							LoopWork() override;
		bool							InternallyTerminated() const override;

	protected:
	public:
		// Constructors & Destructor
										TracerThread(TracerI&);
										TracerThread(const TracerThread&) = delete;
		TracerThread&					operator=(const TracerThread&) = delete;
										~TracerThread();

		// State Change
		void							ChangeCamera(const CameraPerspective&);
		void							ChangeScene(const std::string&);
		void							ChangeResolution(const Vector2ui&);
		void							ChangeTime(double seconds);

		void							ChangeSampleCount(uint32_t);
		void							ChangeParams(const TracerParameters&);
		void							ChangeImageSegment(const Vector2ui& pixelStart,
														   const Vector2ui& pixelCount);
};

inline TracerThread::TracerThread(TracerI& t)
	: tracer(t)
{}

inline TracerThread::~TracerThread()
{
	Stop();
}