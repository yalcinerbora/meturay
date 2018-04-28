#include "TracerThread.h"
#include "TracerI.h"
#include "DistributorI.h"

void TracerThread::THRDLoop(DistributorI& distributor)
{
	// Render Loop
	while(!stopSignal)
	{
		// Check if data is changed
		CameraPerspective newCam;
		Vector2ui newResolution;
		SceneI* newScene;
		uint32_t newSample;
		TracerParameters newParams;
		bool camChanged = camera.CheckChanged(newCam);
		bool sceneChanged = scene.CheckChanged(newScene);
		bool resolutionChanged= resolution.CheckChanged(newResolution);		
		bool sampleChanged = sample.CheckChanged(newSample);
		bool paramsChanged = parameters.CheckChanged(newParams);

		// Reset Image if images changed
		if(resolutionChanged)
			tracer.ResizeImage(newResolution);
		else if(camChanged || sceneChanged)
			tracer.ResetImage();

		if(sceneChanged)
			tracer.AssignScene(*newScene);

		if(paramsChanged)
			tracer.SetParams(newParams);
		

		// Initialize Rays
		// Camera ray generation
		tracer.GenerateCameraRays(newCam, newResolution, newSample);
				
		//=====================//
		//		RenderLoop	   //
		//=====================//
		uint32_t renderCount = 0;
		while(tracer.RayCount() > 0)
		{			
			tracer.HitRays();

			// Check if we are distributed system
			if(!distributor.Alone())
			{
				// Send rays that are not for your responsibility
				tracer.GetMaterialRays();
				distributor.SendMaterialRays();
				
				// Recieve Rays that are responsible
				distributor.RequestMaterialRays();
				tracer.AddMaterialRays();
			}
			
			// We are ready to bounce rays now
			tracer.BounceRays();

		}
		//=====================//
		//	  RenderLoop END   //
		//=====================//

		// Image is ready now send according to the callbacks
		if(distributor.CheckIfRenderRequested(renderCount))
		{
			distributor.SendImage(tracer.GetImage(resolution),


								  ....

								  );
		}
		renderCount++;

		// Check at the end of the loop
		// for the signals
		{
			std::unique_lock<std::mutex> lock(mutex);
			conditionVar.wait(lock, [&]
			{
				return stopSignal || !pauseSignal;
			});
		}
	}

	// Do cleanup

}

// State Change
void TracerThread::ChangeCamera(const CameraPerspective& persp)
{
	camera = persp;
}

void TracerThread::ChangeScene(SceneI& s)
{
	scene = &s;
}

void TracerThread::ChangeResolution(const Vector2ui& res)
{
	resolution = res;
}

void TracerThread::ChangeSampleCount(uint32_t sampleCount)
{
	sample = sampleCount;
}

void TracerThread::ChangeParams(const TracerParameters& p)
{
	parameters = p;
}