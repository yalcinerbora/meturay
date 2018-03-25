#pragma once
/**

Visor Interface

Visor is a standalone program to monitor current 
image that si being rendered by Tracers

VisorView Interface encapsulates rendering window and real-time GPU portion
Visor

*/

#include <cstddef>
#include "Types.h"
#include "Vector.h"

class VisorViewI
{
	public:
		virtual				~VisorViewI() = default;

		// Interface
		virtual bool		IsOpen() = 0;
		virtual void		Present() = 0;

		// Data Related
		virtual void		ResetImageBuffer(const Vector2i& imageSize, PixelFormat) = 0;
		virtual void		SetImagePortion(const Vector2i& start,
											const Vector2i& end,
											const std::byte* data) = 0;
		// Misc
		virtual void		SetWindowSize(const Vector2i& size) = 0;
		virtual void		SetFPSLimit(float) = 0;


		
		// Keyboard I-O (To External Systems)
		//virtual bool		SetSystem() = 0;



};