#pragma once
/**

V

This Distributor is user interfacable thorugh
Visor and Analytic Classes/Programs.

This deleagtes user input and recieves user output (image)

*/

#include <cstdint>
#include <vector>
#include "Vector.h"

struct CameraPerspective;

typedef void(*SetImageSegmentFunc)(const std::vector<Vector3> image,
								   const Vector2ui resolution,
								   const Vector2ui offset,
								   const Vector2ui size);
class VisorNodeI
{

	private:
	protected:
	public:
		virtual					~VisorNodeI() = default;

		// Visor Commands
		virtual void			SetImageStream(bool) = 0;
		virtual void			SetImagePeriod(uint32_t iterations) = 0;

		virtual void			ChangeCamera(const CameraPerspective&) = 0;
		virtual void			ChangeTime(double seconds) = 0;
		virtual void			ChangeFPS(int fps) = 0;
		virtual void			NextFrame() = 0;
		virtual void			PreviousFrame() = 0;

		virtual void			AttachDisplayCallback(SetImageSegmentFunc) = 0;
};