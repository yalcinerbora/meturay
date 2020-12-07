#pragma once

#include "RayLib/TracerStructs.h"

#include "TracerCUDALib/TracerLogicPools.h"

#include "TracerCUDALib/TracerLogicGeneratorI.h"
#include "TracerCUDALib/ScenePartitionerI.h"

//extern "C" _declspec(dllexport) TracerPoolI * __stdcall GenerateBasicTracerPool();
//
//extern "C" _declspec(dllexport) void __stdcall DeleteBasicTracerPool(TracerPoolI*);

//struct TracerSystem
//{
//    const ScenePartitionerI* scenePartitioner;
//    GPUSceneI*;
//};

extern "C" _declspec(dllexport) TracerLogicGeneratorI* __stdcall GenerateTracerSystem(const std::string& scene);

extern "C" _declspec(dllexport) TracerLogicGeneratorI* __stdcall GenerateLogicGenerator();

extern "C" _declspec(dllexport) GPUSceneI* __stdcall GenerateSceneLoader();

extern "C" _declspec(dllexport) void __stdcall DeleteLogicGenerator(TracerLogicGeneratorI*);

extern "C" _declspec(dllexport) void __stdcall DeleteSceneLoader(GPUSceneI*);