#pragma once


class AnimateI
{
	public:
		virtual				~AnimateI() = default;

		// Interface
		virtual void		ChangeFrame(double time) = 0;
};