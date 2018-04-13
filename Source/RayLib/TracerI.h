#pragma once

/**

Tracer Interface

Main Interface for Tracer DLLs. Only GPU tracer will be
implemented, still tracer is interfaced for further implementations

Tracer Interface is a threaded interface (which means that it repsesents a thread)
which does send commands to GPU to do ray tracing 
(it is responsible for utilizing all GPUs on the computer). 



*/

#include <cstdint>

class SceneI;

class TracerI
{
	public:
		virtual					~TracerI() = default;

		// Interface
		virtual void			AssignScene(const SceneI&) = 0;
		
		//
		virtual void			GenerateAccelerator(uint32_t objId) = 0;


		virtual void			Render() = 0;

};
