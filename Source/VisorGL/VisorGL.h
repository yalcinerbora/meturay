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
#include <mutex>

#include "RayLib/VisorI.h"
#include "RayLib/VisorInputI.h"

struct ImagePortion
{
	Vector2i				start;
	Vector2i				end;
	std::vector<Vector3>	data;
};

class VisorGL : public VisorViewI
{
	private:
		//static std::unique_ptr<VisorGL>	instance;
		static VisorGL*				instance;
		
		static constexpr float		PostProcessTriData[6] =
		{
			3.0f, -1.0f,
			-1.0f, 3.0f,
			-1.0f, -1.0f
		};

	private:
		VisorInputI*				input;
		GLFWwindow*					window;
		bool						open;

		// Image portion list
		std::mutex					mutex;
		std::vector<ImagePortion>	portionList;

		//
		GLuint						linearSampler;
		GLuint						texture;

		GLuint						vao;
		GLuint						vBuffer;

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
	//	static VisorGL&			Instance();				
		
		// Interface
		bool					IsOpen() override;
		void					Present() override;
		void					Render() override;

		// Input System
		void					SetInputScheme(VisorInputI*) override;

		// Data Related
		void					ResetImageBuffer(const Vector2i& imageSize, 
												 PixelFormat) override;
		void					SetImagePortion(const Vector2i& start,
												const Vector2i& end,
												const std::vector<Vector3> data) override;
		// Misc
		void					SetWindowSize(const Vector2i&) override;
		void					SetFPSLimit(float) override;
};