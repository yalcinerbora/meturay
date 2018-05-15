#pragma once
/**

Non-distributed version of distributor

It directly delegates a MVisor commands to a single MTracer
In this specific process. (Used for debugging and testing etc.)

*/

#include "VisorDistributorI.h"
#include "TracerDistributorI.h"

class SelfDistributor 
	: public VisorDistributorI
	, public TracerDistributorI
{
	private:
		// Callbacks
		SetCameraFunc			camFunc;
		SetTimeFunc				timeFunc;
		SetParameterFunc		paramFunc;
		StartStopFunc			startStopFunc;
		PauseContFunc			pauseContFunc;
		SetSceneFunc			sceneFunc;

		SetImageSegmentFunc		imgFunc;

		// User Input data
		uint32_t				sendPerIteration;		
		bool					imgStreamOn;

		int						currentFPS;
		int						currentFrame;

	protected:
	public:
		// Constructors & Destructor
								SelfDistributor();
								~SelfDistributor() = default;

		// ================== //
		//  Tracer Interface  //
		// ================== //
		// Sending (All non-blocking)
		void					SendMaterialRays(uint32_t materialId,
														 const RayRecordCPU) override;
		void					SendMaterialRays(const std::vector<ArrayPortion<uint32_t>> materialIds,
														 const RayRecordCPU) override;
		void					SendImage(const std::vector<Vector3f> image,
												  const Vector2ui resolution,
												  const Vector2ui offset = Vector2ui(0, 0),
												  const Vector2ui size = Vector2ui(0, 0)) override;

		// Checking image should be sent
		bool					ShouldSendImage(uint32_t renderCount) override;

		// Requesting & Recieve Callbacks
		// HDD Data
		void					RequestSceneFile(const std::string& fileName) override;
		void					AttachRecieveSceneFileFunc(FileRecieveFunc) override;

		// Memory Data
		void					RequestSceneAccelerator() override;
		void					RequestObjectAccelerator() override;
		void					RequestObjectAccelerator(uint32_t objId) override;
		void					RequestObjectAccelerator(const std::vector<uint32_t>& objIds) override;
		void					AttachAcceleratorCallback(AcceleratorRecieveFunc) override;

		// Waiting (Synchronized operation)
		void					WaitAccelerators() override;
		void					WaitForMaterialRays(RayRecordCPU&) override;

		// Misc.
		uint64_t				TotalCPUMemory() override;
		uint64_t				TotalGPUMemory() override;
		bool					Alone() override;
		
		// Command Callbacks (From Visors)
		void					AttachCameraCallback(SetCameraFunc) override;
		void					AttachTimeCallback(SetTimeFunc) override;
		void					AttachParamCallback(SetParameterFunc) override;
		void					AttachStartStopCallback(StartStopFunc) override;
		void					AttachPauseContCallback(PauseContFunc) override;
		void					AttachSceneCallback(SetSceneFunc) override;

		// ================= //
		//  Visor Interface  //
		// ================= //
		// Visor Commands
		void					SetImageStream(bool) override;
		void					SetImagePeriod(uint32_t iterations) override;

		void					ChangeCamera(const CameraPerspective&) override;
		void					ChangeTime(double seconds) override;
		void					ChangeFPS(int fps) override;
		void					NextFrame() override;
		void					PreviousFrame() override;

		void					AttachDisplayCallback(SetImageSegmentFunc) override;


		// Extra Functionality
		void					SetScene(const std::string& scene);
};