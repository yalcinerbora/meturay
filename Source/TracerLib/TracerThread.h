#pragma once

#include "RayLib/LoopingThreadI.h"
#include "RayLib/ThreadData.h"
#include "RayLib/Camera.h"
#include "RayLib/TracerStructs.h"
#include "RayLib/Types.h"

class SceneI;
class TracerI;
class TracerBaseLogicI;

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
		TracerBaseLogicI&				logic;

		// Current Settings
		ThreadData<double>				time;
		ThreadData<CameraPerspective>	camera;


		ThreadData<Vector2ui>			resolution;
		ThreadData<uint32_t>			sample;
		ThreadData<TracerOptions>		options;

		ThreadData<ImageSegment>		segment;
		ThreadData<PixelFormat>			pixFormat;

		int								currentFPS;
		int								currentFrame;

		// Thread work
		void							InitialWork() override;
		void							LoopWork() override;
		void							FinalWork() override;
		bool							InternallyTerminated() const override;

	protected:
	public:
		// Constructors & Destructor
										TracerThread(TracerI&, TracerBaseLogicI&);
										TracerThread(const TracerThread&) = delete;
		TracerThread&					operator=(const TracerThread&) = delete;
										~TracerThread();

		// State Change
		void							ChangeCamera(const CameraPerspective&);
		void							ChangeResolution(const Vector2ui&);
		void							ChangeTime(double seconds);
		void							ChangePixelFormat(PixelFormat);
		void							ChangeSampleCount(uint32_t);
		void							ChangeOptions(const TracerOptions&);
		void							ChangeImageSegment(const Vector2ui& pixelStart,
														   const Vector2ui& pixelCount);
};

inline TracerThread::TracerThread(TracerI& t, TracerBaseLogicI& l)
	: tracer(t)
	, logic(l)
{}

inline TracerThread::~TracerThread()
{
	Stop();
}