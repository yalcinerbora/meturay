#pragma once

#include "CommandCallbacksI.h"

class VisorCallbacksI : public CommandCallbacksI
{
	public: 
		virtual			~VisorCallbacksI() = default;

		virtual void	WindowMinimizeAction(bool minimized) = 0;
		virtual void	WindowCloseAction() = 0;		
};