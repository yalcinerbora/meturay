#pragma once
/**

Tracer Loader provides functionality
to load tracer logic and its equavilent
sub-logics (material shade functions,
primitive data fetching etc.)

*/


//#ifdef METU_SHARED_TRACERCUDA
//#define METU_SHARED_TRACERCUDA_ENTRY_POINT __declspec(dllexport)
//#else
//#define METU_SHARED_TRACERCUDA_ENTRY_POINT __declspec(dllimport)
//#endif
//
//#include <memory>
//#include "RayLib/TracerI.h"
//
//METU_SHARED_TRACERCUDA_ENTRY_POINT std::unique_ptr<TracerI> CreateTracerCUDA();

#include <memory>
#include "TracerLogicI.h"
#include "TracerBase.h"
#include "TracerThread.h"

#include "RayLib/SharedLibLoader.h"

using LogicInterfaces = std::pair<SharedLibPtr<TracerBaseLogicI>,
								  SharedLibPtr<TracerLogicGeneratorI>>;

namespace TracerLoader
{
	static constexpr const char*	BaseInterfaceConstructorName = "GenBaseTracer";
	static constexpr const char*	BaseInterfaceDestructorName	= "FreeBaseTracer";

	static constexpr const char*	LogicInterfaceConstructorName = "GenLogicGenerator";
	static constexpr const char*	LogicInterfaceDestructorName = "FreeLogicGenerator";

	// Load Tracer Logic From DLL
	// Default named load
	// This should be the usage for tracer
	LogicInterfaces					LoadTracerLogic(SharedLib& s);

	// Custom name load
	// This is provided for test tracer
	// Sine different test cases may need different tracer implementations
	// This makes many different tracers to reside ona single dll
	// This may be usefull
	LogicInterfaces				LoadTracerLogic(SharedLib& s,
												const char* baseConst,
												const char* baseDest,
												const char* logicConst,
												const char* logicDest);

	TracerThread				GenerateTracerThread(TracerBase&, 
													 TracerBaseLogicI&, 
													 uint32_t seed);
}