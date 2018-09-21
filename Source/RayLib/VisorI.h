#pragma once
/**

Visor Interface

Visor is a standalone program to monitor current 
image that si being rendered by Tracers

VisorView Interface encapsulates rendering window and real-time GPU portion
Visor

*/

#include <cstddef>
#include <vector>
#include "Types.h"
#include "Vector.h"

class VisorInputI;

struct VisorOptions
{
	// Technical
	size_t				eventBufferSize;

	// Window Related
	bool				stereoOn;
	PixelFormat			iFormat;
	Vector2i			iSize;
};

template <class T>
class VisorSystemI
{
	public:
		virtual				~VisorSystemI() = default;

		// Interface
		virtual void		CreateSystem() = 0;
		virtual void		TerminateSystem() = 0;
		
		//T					CreateWindow(VisorOptions);
		//void				DestroyWindow(T&);
};

class VisorViewI
{
	public:
		virtual				~VisorViewI() = default;

		// Interface
		virtual bool		IsOpen() = 0;		
		virtual void		Render() = 0;
		virtual void		ProcessInputs() = 0;
		// Input System
		virtual void		SetInputScheme(VisorInputI*) = 0;

		// Data Related
		virtual void		ResetImageBuffer(const Vector2i& imageSize, PixelFormat) = 0;
		virtual void		SetImagePortion(const Vector2i& start,
											const Vector2i& end,
											const std::vector<float> data) = 0;
		// Misc
		virtual void		SetWindowSize(const Vector2i& size) = 0;
		virtual void		SetFPSLimit(float) = 0;
};