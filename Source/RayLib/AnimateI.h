#pragma once

struct Error;

class AnimateI
{
	public:
		virtual				~AnimateI() = default;

		// Interface
		virtual Error		ChangeFrame(double time) = 0;
};