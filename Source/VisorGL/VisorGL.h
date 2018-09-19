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

#include "ShaderGL.h"

// Basic command list implementation
struct VisorGLCommand
{
	public:
		enum Type
		{
			SET_PORTION,
			RESET_IMAGE
		};

	public:
		Type type;

		// Data will be
		PixelFormat				format;
		Vector2i				startOrSize;
		Vector2i				end;
		std::vector<float>		data;

		// Commands should not be copied
		VisorGLCommand() = default;
		VisorGLCommand(VisorGLCommand&&) = default;
		VisorGLCommand(const VisorGLCommand&) = delete;
		VisorGLCommand& operator=(const VisorGLCommand&) = delete;
		VisorGLCommand& operator=(VisorGLCommand&&) = default;
};

class VisorGL : public VisorViewI
{
	private:	
		static VisorGL*				instance;
		
		static constexpr float		PostProcessTriData[6] =
		{
			3.0f, -1.0f,
			-1.0f, 3.0f,
			-1.0f, -1.0f
		};

		// Shader Location Cnstants
		// T: Texture Object
		// IN: Shader Inputs
		// OUT: SHader Outputs
		static constexpr GLenum		T_INPUT = 0;
		static constexpr GLenum		IN_POS = 0;

	private:
		VisorInputI*				input;
		GLFWwindow*					window;
		bool						open;

		// Image portion list
		std::mutex					mutexCommand;
		std::vector<VisorGLCommand>	commandList;
		Vector2i					viewportSize;

		// Image Texture		
		GLuint						linearSampler;
		GLuint						texture;
		PixelFormat					texPixFormat;

		// Shader
		ShaderGL					vertPP;
		ShaderGL					fragPP;

		// Vertex
		GLuint						vao;
		GLuint						vBuffer;

		static KeyAction			DetermineAction(int);
		static MouseButtonType		DetermineMouseButton(int);
		static KeyboardKeyType		DetermineKey(int);

		// Callbacks
		// GLFW
		static void					ErrorCallbackGLFW(int, const char*);
		static void					WindowPosGLFW(GLFWwindow*, int, int);
		static void					WindowFBGLFW(GLFWwindow*, int, int);
		static void					WindowSizeGLFW(GLFWwindow*, int, int);
		static void					WindowCloseGLFW(GLFWwindow*);
		static void					WindowRefreshGLFW(GLFWwindow*);
		static void					WindowFocusedGLFW(GLFWwindow*, int);
		static void					WindowMinimizedGLFW(GLFWwindow*, int);

		static void					KeyboardUsedGLFW(GLFWwindow*, int, int, int, int);
		static void					MouseMovedGLFW(GLFWwindow*, double, double);
		static void					MousePressedGLFW(GLFWwindow*, int, int, int);
		static void					MouseScrolledGLFW(GLFWwindow*, double, double);

		// OGL Debug Context Callback
		static void __stdcall		OGLCallbackRender(GLenum source,
													  GLenum type,
													  GLuint id,
													  GLenum severity,
													  GLsizei length,
													  const char* message,
													  const void* userParam);
		// Visor to OGL conversions
		static GLenum				PixelFormatToGL(PixelFormat);
		static GLenum				PixelFormatToSizedGL(PixelFormat);

		// Internal Command Handling
		void						ProcessCommand(const VisorGLCommand&);
		void						RenderImage();

	protected:
	public:
		// Constructors & Destructor
								VisorGL(const VisorOptions&);
								VisorGL(const VisorGL&) = delete;
		VisorGL&				operator=(const VisorGL&) = delete;
								~VisorGL();
		
		// Interface
		bool					IsOpen() override;
		void					Present() override;
		void					Render() override;

		// Input System
		void					SetInputScheme(VisorInputI*) override;

		// Data Related (Any Thread Callable
		void					ResetImageBuffer(const Vector2i& imageSize, 
												 PixelFormat) override;
		void					SetImagePortion(const Vector2i& start,
												const Vector2i& end,
												const std::vector<float> data) override;
		// Misc (Only main thread callable)
		void					SetWindowSize(const Vector2i& size) override;
		void					SetFPSLimit(float) override;
};