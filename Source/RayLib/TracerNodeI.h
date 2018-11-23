#pragma once

/**

Tracer Distributor Interface.

This interface defines process behaviour between


*/

#include <cstdint>
#include <vector>

#include "Vector.h"
#include "ArrayPortion.h"
#include "SceneStructs.h"

struct CameraPerspective;
struct TracerParameters;
struct MatBatchRayDataCPU;
enum class ErrorType;

enum AcceleratorType
{
	BVH_SCENE,
	BVH_OBJECT,
	SVO_OBJECT
};

typedef void(*FileRecieveFunc)(const std::string fileName,
							   const std::vector<char> fileData);

typedef void(*AcceleratorRecieveFunc)(const uint32_t objId,
									  const AcceleratorType,
									  const std::vector<char>);

typedef void(*SetSceneFunc)(const std::string);
typedef void(*SetCameraFunc)(const CameraPerspective);
typedef void(*SetTimeFunc)(const double);
typedef void(*SetParameterFunc)(const TracerParameters);
typedef void(*StartStopFunc)(const bool);
typedef void(*PauseContFunc)(const bool);

class TracerNodeI
{
	public:
		virtual					~TracerNodeI() = default;

		// Static Initialization Time Data

		// Sending (All non-blocking)
		virtual void			SendMaterialRays(const std::vector<MatBatchRayDataCPU> matRayData) = 0;
		virtual void			SendImage(const std::vector<Vector3f> image,
										  const Vector2ui resolution,
										  const Vector2ui offset = Vector2ui(0, 0),
										  const Vector2ui size = Vector2ui(0, 0)) = 0;
		virtual void			SendError(uint32_t errorEnum, ErrorType) = 0;

		// Checking image should be sent
		virtual bool			ShouldSendImage(uint32_t renderCount) = 0;

		// Requesting & Recieve Callbacks
		// HDD Data
		virtual void			RequestSceneFile(const std::string& fileName) = 0;
		virtual void			AttachRecieveSceneFileFunc(FileRecieveFunc) = 0;

		// Memory Data
		virtual void			RequestSceneAccelerator() = 0;
		virtual void			RequestObjectAccelerator() = 0;
		virtual void			RequestObjectAccelerator(uint32_t objId) = 0;
		virtual void			RequestObjectAccelerator(const std::vector<uint32_t>& objIds) = 0;
		virtual void			AttachAcceleratorCallback(AcceleratorRecieveFunc) = 0;

		// Waiting (Synchronized operation)
		virtual void			WaitAccelerators() = 0;
		virtual void			WaitForMaterialRays(std::vector<MatBatchRayDataCPU>&) = 0;

		// Misc.
		virtual uint64_t		TotalCPUMemory() = 0;
		virtual uint64_t		TotalGPUMemory() = 0;
		virtual uint64_t		GPUMemory(int GPUId) = 0;
		virtual bool			Alone() = 0;

		// Command Callbacks (From Visors)
		virtual void			AttachCameraCallback(SetCameraFunc) = 0;
		virtual void			AttachTimeCallback(SetTimeFunc) = 0;
		virtual void			AttachParamCallback(SetParameterFunc) = 0;
		virtual void			AttachStartStopCallback(StartStopFunc) = 0;
		virtual void			AttachPauseContCallback(PauseContFunc) = 0;
		virtual void			AttachSceneCallback(SetSceneFunc) = 0;		
};
