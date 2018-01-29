#pragma once

/**

Tracer Interface

Main Interface for Tracer DLLs. Only single GPU tracer will be
implemented, still tracer is interfaced for further implementations

*/

class SceneI;

class TracerI
{
	public:
		virtual					~TracerI() = default;

		// Interface	
		virtual void			Render(const SceneI&) = 0;
};
