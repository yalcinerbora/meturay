#pragma once
/**

OGL Implementation of Visor View
Uses GLFW glfw has c style interface and required to be initalized 
at start of the program. We will need single window thus making
the VisorGL singleton.


*/

#include <gl\glew.h>
#include <glfw/glfw3.h>
#include <memory>

#include "RayLib/VisorI.h"
#include "RayLib/VisorInputI.h"

class VisorGL : public VisorViewI
{
	private:
		static std::unique_ptr<VisorGL>	instance;

	private:
		VisorInputI*		input;
		GLFWwindow*			window;
		bool				open;

		
		


		static KeyAction		DetermineAction(int);
		static MouseButtonType	DetermineMouseButton(int);
		static KeyboardKeyType	DetermineKey(int);

		// Callbacks
		// GLFW
		static void				ErrorCallbackGLFW(int, const char*);
		static void				WindowPosGLFW(GLFWwindow*, int, int);
		static void				WindowFBGLFW(GLFWwindow*, int, int);
		static void				WindowSizeGLFW(GLFWwindow*, int, int);
		static void				WindowCloseGLFW(GLFWwindow*);
		static void				WindowRefreshGLFW(GLFWwindow*);
		static void				WindowFocusedGLFW(GLFWwindow*, int);
		static void				WindowMinimizedGLFW(GLFWwindow*, int);

		static void				KeyboardUsedGLFW(GLFWwindow*, int, int, int, int);
		static void				MouseMovedGLFW(GLFWwindow*, double, double);
		static void				MousePressedGLFW(GLFWwindow*, int, int, int);
		static void				MouseScrolledGLFW(GLFWwindow*, double, double);

		// OGL Debug Context Callback
		static void __stdcall	OGLCallbackRender(GLenum source,
												  GLenum type,
												  GLuint id,
												  GLenum severity,
												  GLsizei length,
												  const char* message,
												  const void* userParam);

	protected:
	public:
		// Constructors & Destructor
								VisorGL();
								VisorGL(const VisorGL&) = delete;
		VisorGL&				operator=(const VisorGL&) = delete;
								~VisorGL();

		// Singleton Accessor 
		static VisorGL&			Instance();
		
		// Interface
		bool					IsOpen() override;
		void					Present() override;

		// Input System
		void					SetInputScheme(VisorInputI*) override;

		// Data Related
		void					ResetImageBuffer(const Vector2i& imageSize, 
												 PixelFormat) override;
		void					SetImagePortion(const Vector2i& start,
												const Vector2i& end,
												const std::byte* data) override;
		// Misc
		void					SetWindowSize(const Vector2i&) override;
		void					SetFPSLimit(float) override;
};