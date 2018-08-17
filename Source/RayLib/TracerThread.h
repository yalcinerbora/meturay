#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>

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
class TracerThread
{
	private:
		struct ImageSegment
		{
			Vector2ui pixelStart;
			Vector2ui pixelCount;
		};

	private:
		// Threading and thread management
		std::thread						thread;
		std::mutex						mutex;
		std::condition_variable			conditionVar;
		bool							stopSignal;
		bool							pauseSignal;

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

		// Actual Thread Loop
		void							THRDLoop(TracerDistributorI&, uint32_t seed);

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
		// Render Loop Related
		void							Start(TracerDistributorI&, uint32_t seed = 0);
		void							Stop();
		void							Pause(bool);
};

inline TracerThread::TracerThread(TracerI& t)
	: tracer(t)
	, pauseSignal(false)
	, stopSignal(false)
{}

inline TracerThread::~TracerThread()
{
	Stop();
}

inline void TracerThread::Start(TracerDistributorI& d, uint32_t seed)
{
	thread = std::thread(&TracerThread::THRDLoop, this, std::ref(d), seed);
}

inline void TracerThread::Stop()
{
	mutex.lock();
	stopSignal = true;
	mutex.unlock();
	conditionVar.notify_one();
	if(thread.joinable()) thread.join();
	stopSignal = false;
}

inline void TracerThread::Pause(bool pause)
{
	mutex.lock();
	pauseSignal = pause;
	mutex.unlock();
	conditionVar.notify_one();
}
