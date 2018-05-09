#pragma once


class AnimateI
{
	public:
		virtual				~AnimateI() = default;

		// Interface
		virtual void		ChangeTime(double time) = 0;
};