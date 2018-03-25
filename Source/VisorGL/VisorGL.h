#pragma once
/**

OGL Implementation of Visor View

*/

#include <gl\glew.h>
#include <glfw/glfw3.h>

#include "RayLib/VisorI.h"

class VisorGL : public VisorViewI
{
	private:
		GLFWwindow*			window;

		bool				open;

		// Callbacks
		// GLFW
		static void				ErrorCallbackGLFW(int, const char*);

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
							VisorGL(VisorGL&&);
							VisorGL(const VisorGL&) = delete;
							VisorGL& operator=(VisorGL&&);
							VisorGL& operator=(const VisorGL&) = delete;
							~VisorGL();

		// Interface
		bool				IsOpen() override;
		void				Present() override;

		// Data Related
		void				ResetImageBuffer(const Vector2i& imageSize, PixelFormat) override;
		void				SetImagePortion(const Vector2i& start,
											const Vector2i& end,
											const std::byte* data) override;
		// Misc
		void				SetWindowSize(const Vector2i&) override;
		void				SetFPSLimit(float) override;
	

};