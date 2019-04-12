#pragma once
/**






*/


#include "RayLib/VisorInputI.h"
#include "RayLib/Camera.h"
#include "ThreadData.h"

class VisorWindowInput : public VisorInputI
{
	private:
		static constexpr int			PredefinedFPS[] = {24, 30, 60, 90, 120, 240};
		static constexpr unsigned int	FPSCount = (sizeof(PredefinedFPS) / sizeof(int));

		// Camera Movement portion
		const double					Sensitivity;
		const double					MoveRatio;
		const double					MoveRatioModifier;

		bool							fpsMode;
		double							mouseX;
		double							mouseY;
		double							currentRatio;


		// Scene Times
		unsigned int					currentFPS;


		CameraPerspective				camera;
		VisorCallbacksI*				visorCallbacks;

	protected:
	public:
		// Constructor & Destructor
								VisorWindowInput(double sensitivity,
												 double moveRatio,
												 double moveRatioModifier);
								~VisorWindowInput() = default;

		// Implementation		
		void					AttachVisorCallback(VisorCallbacksI&) override;

		void					WindowPosChanged(int posX, int posY) override;
		void					WindowFBChanged(int fbWidth, int fbHeight) override;
		void					WindowSizeChanged(int width, int height) override;
		void					WindowClosed() override;
		void					WindowRefreshed() override;
		void					WindowFocused(bool) override;
		void					WindowMinimized(bool) override;

		void					MouseScrolled(double xOffset, double yOffset) override;
		void					MouseMoved(double x, double y) override;

		void					KeyboardUsed(KeyboardKeyType key, KeyAction action) override;
		void					MouseButtonUsed(MouseButtonType button, KeyAction action) override;
};