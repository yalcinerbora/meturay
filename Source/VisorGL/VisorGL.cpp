#include "VisorGL.h"
#include "RayLib/Log.h"

void VisorGL::ErrorCallbackGLFW(int error, const char* description)
{
	METU_ERROR_LOG("GLFW Error %d: %s", error, description);
}

void __stdcall VisorGL::OGLCallbackRender(GLenum,
										  GLenum type,
										  GLuint id,
										  GLenum severity,
										  GLsizei,
										  const char* message,
										  const void*)
{
	// Dont Show Others For Now
	if(type == GL_DEBUG_TYPE_OTHER ||	//
	   id == 131186 ||					// Buffer Copy warning omit
	   id == 131218)					// Shader recompile cuz of state mismatch omit
		return;

	METU_DEBUG_LOG("---------------------OGL-Callback-Render------------");
	METU_DEBUG_LOG("Message: %s", message);
	switch(type)
	{
		case GL_DEBUG_TYPE_ERROR:
			METU_DEBUG_LOG("Type: ERROR");
			break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			METU_DEBUG_LOG("Type: DEPRECATED_BEHAVIOR");
			break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			METU_DEBUG_LOG("Type: UNDEFINED_BEHAVIOR");
			break;
		case GL_DEBUG_TYPE_PORTABILITY:
			METU_DEBUG_LOG("Type: PORTABILITY");
			break;
		case GL_DEBUG_TYPE_PERFORMANCE:
			METU_DEBUG_LOG("Type: PERFORMANCE");
			break;
		case GL_DEBUG_TYPE_OTHER:
			METU_DEBUG_LOG("Type: OTHER");
			break;
	}

	METU_DEBUG_LOG("ID: %d", id);
	switch(severity)
	{
		case GL_DEBUG_SEVERITY_LOW:
			METU_DEBUG_LOG("Severity: LOW");
			break;
		case GL_DEBUG_SEVERITY_MEDIUM:
			METU_DEBUG_LOG("Severity: MEDIUM");
			break;
		case GL_DEBUG_SEVERITY_HIGH:
			METU_DEBUG_LOG("Severity: HIGH");
			break;
		default:
			METU_DEBUG_LOG("Severity: NONE");
			break;
	}
	METU_DEBUG_LOG("---------------------OGL-Callback-Render-End--------------");
}

VisorGL::VisorGL()
{
	if(!glfwInit())
	{
		METU_ERROR_LOG("Could not Init GLFW");
		assert(false);
	}
	glfwSetErrorCallback(ErrorCallbackGLFW);

	// Common Window Hints
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
	glfwWindowHint(GLFW_SRGB_CAPABLE, GL_FALSE);	// Buggy

	glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_NONE);

	// Debug Context
	if constexpr(IS_DEBUG_MODE)
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	else
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_FALSE);

	// At most 16x MSAA
	glfwWindowHint(GLFW_SAMPLES, 16);

	// Pixel of WindowFBO

	// Full precision output 
	glfwWindowHint(GLFW_RED_BITS, 32);
	glfwWindowHint(GLFW_GREEN_BITS, 32);
	glfwWindowHint(GLFW_BLUE_BITS, 32);

	// No depth buffer or stencil buffer etc
	glfwWindowHint(GLFW_ALPHA_BITS, 0);
	glfwWindowHint(GLFW_DEPTH_BITS, 0);
	glfwWindowHint(GLFW_STENCIL_BITS, 0);

	window = glfwCreateWindow(1280, 720, "METU Visor", nullptr, nullptr);
	if(window == nullptr)
	{
		METU_ERROR_LOG("Error: Could not create window.");
		assert(false);
	}
	glfwMakeContextCurrent(window);

	// Now Init GLEW
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if(err != GLEW_OK)
	{
		METU_ERROR_LOG("Error: %s\n", glewGetErrorString(err));
		assert(false);
	}

	// Print Stuff Now
	// Window Done
	METU_LOG("Window Initialized.");
	METU_LOG("GLEW\t: %s", glewGetString(GLEW_VERSION));
	METU_LOG("GLFW\t: %s", glfwGetVersionString());
	METU_LOG("");
	METU_LOG("Renderer Information...");
	METU_LOG("OpenGL\t: %s", glGetString(GL_VERSION));
	METU_LOG("GLSL\t: %s", glGetString(GL_SHADING_LANGUAGE_VERSION));
	METU_LOG("Device\t: %s", glGetString(GL_RENDERER));
	METU_LOG("");

	if constexpr (IS_DEBUG_MODE)
	{
		// Add Callback
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(VisorGL::OGLCallbackRender, nullptr);
		glDebugMessageControl(GL_DONT_CARE,
							  GL_DONT_CARE,
							  GL_DONT_CARE,
							  0,
							  nullptr,
							  GL_TRUE);
	}
}

VisorGL::VisorGL(VisorGL&& other)
{

}

VisorGL& VisorGL::operator=(VisorGL&& other)
{
	return *this;
}

VisorGL::~VisorGL()
{
	if(window != nullptr) glfwDestroyWindow(window);
	glfwTerminate();
}

bool VisorGL::IsOpen()
{
	return open;
}

void VisorGL::Present()
{
	glfwPollEvents();
}

void VisorGL::ResetImageBuffer(const Vector2i& imageSize, PixelFormat)
{
	glfwSetWindowSize(window, imageSize[0], imageSize[1]);
	//....
	glfwShowWindow(window);
}

void VisorGL::SetImagePortion(const Vector2i& start,
							  const Vector2i& end,
							  const std::byte* data)
{
	
}

void VisorGL::SetWindowSize(const Vector2i&)
{

}

void VisorGL::SetFPSLimit(float f)
{

}