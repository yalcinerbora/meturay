#include "VisorGL.h"
#include "RayLib/Log.h"
#include <map>
#include <cassert>

KeyAction VisorGL::DetermineAction(int action)
{
	if(action == GLFW_PRESS)
	{
		return KeyAction::PRESSED;
	}
	else if(action == GLFW_RELEASE)
	{
		return KeyAction::RELEASED;
	}
	else if( action == GLFW_REPEAT)
	{
		return KeyAction::REPEATED;
	}
	else
	{
		assert(false);
		return KeyAction::PRESSED;			
	}
		
}

MouseButtonType VisorGL::DetermineMouseButton(int button)
{
	static std::map<int, MouseButtonType> buttonMap =
	{
		std::make_pair(GLFW_MOUSE_BUTTON_LEFT, MouseButtonType::LEFT),
		std::make_pair(GLFW_MOUSE_BUTTON_RIGHT, MouseButtonType::RIGHT),
		std::make_pair(GLFW_MOUSE_BUTTON_MIDDLE, MouseButtonType::MIDDLE),
		std::make_pair(GLFW_MOUSE_BUTTON_4, MouseButtonType::BUTTON_4),
		std::make_pair(GLFW_MOUSE_BUTTON_5, MouseButtonType::BUTTON_5),
		std::make_pair(GLFW_MOUSE_BUTTON_6, MouseButtonType::BUTTON_6),
		std::make_pair(GLFW_MOUSE_BUTTON_7, MouseButtonType::BUTTON_7),
		std::make_pair(GLFW_MOUSE_BUTTON_8, MouseButtonType::BUTTON_8)
	};
	return buttonMap[button];
}

KeyboardKeyType VisorGL::DetermineKey(int key)
{
	static std::map<int, KeyboardKeyType> keyMap =
	{
		std::make_pair(GLFW_KEY_SPACE, KeyboardKeyType::SPACE),
		std::make_pair(GLFW_KEY_APOSTROPHE, KeyboardKeyType::APOSTROPHE),
		std::make_pair(GLFW_KEY_COMMA, KeyboardKeyType::COMMA),
		std::make_pair(GLFW_KEY_MINUS, KeyboardKeyType::MINUS),
		std::make_pair(GLFW_KEY_PERIOD, KeyboardKeyType::PERIOD),
		std::make_pair(GLFW_KEY_SLASH, KeyboardKeyType::SLASH),
		std::make_pair(GLFW_KEY_0, KeyboardKeyType::NUMBER_0),
		std::make_pair(GLFW_KEY_1, KeyboardKeyType::NUMBER_1),
		std::make_pair(GLFW_KEY_2, KeyboardKeyType::NUMBER_2),
		std::make_pair(GLFW_KEY_3, KeyboardKeyType::NUMBER_3),
		std::make_pair(GLFW_KEY_4, KeyboardKeyType::NUMBER_4),
		std::make_pair(GLFW_KEY_5, KeyboardKeyType::NUMBER_5),
		std::make_pair(GLFW_KEY_6, KeyboardKeyType::NUMBER_6),
		std::make_pair(GLFW_KEY_7, KeyboardKeyType::NUMBER_7),
		std::make_pair(GLFW_KEY_8, KeyboardKeyType::NUMBER_8),
		std::make_pair(GLFW_KEY_9, KeyboardKeyType::NUMBER_9),
		std::make_pair(GLFW_KEY_SEMICOLON, KeyboardKeyType::SEMICOLON),
		std::make_pair(GLFW_KEY_EQUAL, KeyboardKeyType::EQUAL),
		std::make_pair(GLFW_KEY_A, KeyboardKeyType::A),
		std::make_pair(GLFW_KEY_B, KeyboardKeyType::B),
		std::make_pair(GLFW_KEY_C, KeyboardKeyType::C),
		std::make_pair(GLFW_KEY_D, KeyboardKeyType::D),
		std::make_pair(GLFW_KEY_E, KeyboardKeyType::E),
		std::make_pair(GLFW_KEY_F, KeyboardKeyType::F),
		std::make_pair(GLFW_KEY_G, KeyboardKeyType::G),
		std::make_pair(GLFW_KEY_H, KeyboardKeyType::H),
		std::make_pair(GLFW_KEY_I, KeyboardKeyType::I),
		std::make_pair(GLFW_KEY_J, KeyboardKeyType::J),
		std::make_pair(GLFW_KEY_K, KeyboardKeyType::K),
		std::make_pair(GLFW_KEY_L, KeyboardKeyType::L),
		std::make_pair(GLFW_KEY_M, KeyboardKeyType::M),
		std::make_pair(GLFW_KEY_N, KeyboardKeyType::N),
		std::make_pair(GLFW_KEY_O, KeyboardKeyType::O),
		std::make_pair(GLFW_KEY_P, KeyboardKeyType::P),
		std::make_pair(GLFW_KEY_Q, KeyboardKeyType::Q),
		std::make_pair(GLFW_KEY_R, KeyboardKeyType::R),
		std::make_pair(GLFW_KEY_S, KeyboardKeyType::S),
		std::make_pair(GLFW_KEY_T, KeyboardKeyType::T),
		std::make_pair(GLFW_KEY_U, KeyboardKeyType::U),
		std::make_pair(GLFW_KEY_V, KeyboardKeyType::V),
		std::make_pair(GLFW_KEY_W, KeyboardKeyType::W),
		std::make_pair(GLFW_KEY_X, KeyboardKeyType::X),
		std::make_pair(GLFW_KEY_Y, KeyboardKeyType::Y),
		std::make_pair(GLFW_KEY_Z, KeyboardKeyType::Z),
		std::make_pair(GLFW_KEY_LEFT_BRACKET, KeyboardKeyType::LEFT_BRACKET),
		std::make_pair(GLFW_KEY_BACKSLASH, KeyboardKeyType::BACKSLASH),
		std::make_pair(GLFW_KEY_RIGHT_BRACKET, KeyboardKeyType::RIGHT_BRACKET),
		std::make_pair(GLFW_KEY_GRAVE_ACCENT, KeyboardKeyType::GRAVE_ACCENT),
		std::make_pair(GLFW_KEY_WORLD_1, KeyboardKeyType::WORLD_1),
		std::make_pair(GLFW_KEY_WORLD_2, KeyboardKeyType::WORLD_2),
		std::make_pair(GLFW_KEY_ESCAPE, KeyboardKeyType::ESCAPE),
		std::make_pair(GLFW_KEY_ENTER, KeyboardKeyType::ENTER),
		std::make_pair(GLFW_KEY_TAB, KeyboardKeyType::TAB),
		std::make_pair(GLFW_KEY_BACKSPACE, KeyboardKeyType::BACKSPACE),
		std::make_pair(GLFW_KEY_INSERT, KeyboardKeyType::INSERT),
		std::make_pair(GLFW_KEY_DELETE, KeyboardKeyType::DELETE_KEY),
		std::make_pair(GLFW_KEY_RIGHT, KeyboardKeyType::RIGHT),
		std::make_pair(GLFW_KEY_LEFT, KeyboardKeyType::LEFT),
		std::make_pair(GLFW_KEY_DOWN, KeyboardKeyType::DOWN),
		std::make_pair(GLFW_KEY_UP, KeyboardKeyType::UP),
		std::make_pair(GLFW_KEY_PAGE_UP, KeyboardKeyType::PAGE_UP),
		std::make_pair(GLFW_KEY_PAGE_DOWN, KeyboardKeyType::PAGE_DOWN),
		std::make_pair(GLFW_KEY_HOME, KeyboardKeyType::HOME),
		std::make_pair(GLFW_KEY_END, KeyboardKeyType::END),
		std::make_pair(GLFW_KEY_CAPS_LOCK, KeyboardKeyType::CAPS_LOCK),
		std::make_pair(GLFW_KEY_SCROLL_LOCK, KeyboardKeyType::SCROLL_LOCK),
		std::make_pair(GLFW_KEY_NUM_LOCK, KeyboardKeyType::NUM_LOCK),
		std::make_pair(GLFW_KEY_PRINT_SCREEN, KeyboardKeyType::PRINT_SCREEN),
		std::make_pair(GLFW_KEY_PAUSE, KeyboardKeyType::PAUSE),
		std::make_pair(GLFW_KEY_F1, KeyboardKeyType::F1),
		std::make_pair(GLFW_KEY_F2, KeyboardKeyType::F2),
		std::make_pair(GLFW_KEY_F3, KeyboardKeyType::F3),
		std::make_pair(GLFW_KEY_F4, KeyboardKeyType::F4),
		std::make_pair(GLFW_KEY_F5, KeyboardKeyType::F5),
		std::make_pair(GLFW_KEY_F6, KeyboardKeyType::F6),
		std::make_pair(GLFW_KEY_F7, KeyboardKeyType::F7),
		std::make_pair(GLFW_KEY_F8, KeyboardKeyType::F8),
		std::make_pair(GLFW_KEY_F9, KeyboardKeyType::F9),
		std::make_pair(GLFW_KEY_F10, KeyboardKeyType::F10),
		std::make_pair(GLFW_KEY_F11, KeyboardKeyType::F11),
		std::make_pair(GLFW_KEY_F12, KeyboardKeyType::F12),
		std::make_pair(GLFW_KEY_F13, KeyboardKeyType::F13),
		std::make_pair(GLFW_KEY_F14, KeyboardKeyType::F14),
		std::make_pair(GLFW_KEY_F15, KeyboardKeyType::F15),
		std::make_pair(GLFW_KEY_F16, KeyboardKeyType::F16),
		std::make_pair(GLFW_KEY_F17, KeyboardKeyType::F17),
		std::make_pair(GLFW_KEY_F18, KeyboardKeyType::F18),
		std::make_pair(GLFW_KEY_F19, KeyboardKeyType::F19),
		std::make_pair(GLFW_KEY_F20, KeyboardKeyType::F20),
		std::make_pair(GLFW_KEY_F21, KeyboardKeyType::F21),
		std::make_pair(GLFW_KEY_F22, KeyboardKeyType::F22),
		std::make_pair(GLFW_KEY_F23, KeyboardKeyType::F23),
		std::make_pair(GLFW_KEY_F24, KeyboardKeyType::F24),
		std::make_pair(GLFW_KEY_F25, KeyboardKeyType::F25),
		std::make_pair(GLFW_KEY_KP_0, KeyboardKeyType::KP_0),
		std::make_pair(GLFW_KEY_KP_1, KeyboardKeyType::KP_1),
		std::make_pair(GLFW_KEY_KP_2, KeyboardKeyType::KP_2),
		std::make_pair(GLFW_KEY_KP_3, KeyboardKeyType::KP_3),
		std::make_pair(GLFW_KEY_KP_4, KeyboardKeyType::KP_4),
		std::make_pair(GLFW_KEY_KP_5, KeyboardKeyType::KP_5),
		std::make_pair(GLFW_KEY_KP_6, KeyboardKeyType::KP_6),
		std::make_pair(GLFW_KEY_KP_7, KeyboardKeyType::KP_7),
		std::make_pair(GLFW_KEY_KP_8, KeyboardKeyType::KP_8),
		std::make_pair(GLFW_KEY_KP_9, KeyboardKeyType::KP_9),
		std::make_pair(GLFW_KEY_KP_DECIMAL, KeyboardKeyType::KP_DECIMAL),
		std::make_pair(GLFW_KEY_KP_DIVIDE, KeyboardKeyType::KP_DIVIDE),
		std::make_pair(GLFW_KEY_KP_MULTIPLY, KeyboardKeyType::KP_MULTIPLY),
		std::make_pair(GLFW_KEY_KP_SUBTRACT, KeyboardKeyType::KP_SUBTRACT),
		std::make_pair(GLFW_KEY_KP_ADD, KeyboardKeyType::KP_ADD),
		std::make_pair(GLFW_KEY_KP_ENTER, KeyboardKeyType::KP_ENTER),
		std::make_pair(GLFW_KEY_KP_EQUAL, KeyboardKeyType::EQUAL),
		std::make_pair(GLFW_KEY_LEFT_SHIFT, KeyboardKeyType::LEFT_SHIFT),
		std::make_pair(GLFW_KEY_LEFT_CONTROL, KeyboardKeyType::LEFT_CONTROL),
		std::make_pair(GLFW_KEY_LEFT_ALT, KeyboardKeyType::LEFT_ALT),
		std::make_pair(GLFW_KEY_LEFT_SUPER, KeyboardKeyType::LEFT_SUPER),
		std::make_pair(GLFW_KEY_RIGHT_SHIFT, KeyboardKeyType::RIGHT_SHIFT),
		std::make_pair(GLFW_KEY_RIGHT_CONTROL, KeyboardKeyType::RIGHT_CONTROL),
		std::make_pair(GLFW_KEY_RIGHT_ALT, KeyboardKeyType::RIGHT_ALT),
		std::make_pair(GLFW_KEY_RIGHT_SUPER, KeyboardKeyType::RIGHT_SUPER),
		std::make_pair(GLFW_KEY_MENU, KeyboardKeyType::MENU)
	};
	return keyMap[key];
}

void VisorGL::ErrorCallbackGLFW(int error, const char* description)
{
	METU_ERROR_LOG("GLFW Error %d: %s", error, description);
}

void VisorGL::WindowPosGLFW(GLFWwindow* w, int x, int y)
{
	assert(instance->window == w);
	if(instance->input) instance->input->WindowPosChanged(x, y);
}

void VisorGL::WindowFBGLFW(GLFWwindow* w, int width, int height)
{
	assert(instance->window == w);
	if(instance->input) instance->input->WindowFBChanged(width, height);
}

void VisorGL::WindowSizeGLFW(GLFWwindow* w, int width, int height)
{
	assert(instance->window == w);
	if(instance->input) instance->input->WindowSizeChanged(width, height);
}

void VisorGL::WindowCloseGLFW(GLFWwindow* w)
{
	assert(instance->window == w);
	if(instance->input) instance->input->WindowClosed();
	instance->open = false;
}

void VisorGL::WindowRefreshGLFW(GLFWwindow* w)
{
	assert(instance->window == w);
	if(instance->input) instance->input->WindowRefreshed();
}

void VisorGL::WindowFocusedGLFW(GLFWwindow* w, int b)
{
	assert(instance->window == w);
	if(instance->input) instance->input->WindowFocused(b);
}

void VisorGL::WindowMinimizedGLFW(GLFWwindow* w, int b)
{
	assert(instance->window == w);
	if(instance->input) instance->input->WindowMinimized(b);
}

void VisorGL::KeyboardUsedGLFW(GLFWwindow* w, int key, int scanCode, 
							   int action, int modifiers)
{
	assert(instance->window == w);
	if(instance->input) instance->input->KeyboardUsed(DetermineKey(key), DetermineAction(action));
}

void VisorGL::MouseMovedGLFW(GLFWwindow* w, double x, double y)
{
	assert(instance->window == w);
	if(instance->input) instance->input->MouseMoved(x, y);

}

void VisorGL::MousePressedGLFW(GLFWwindow* w, int button, int action, int modifier)
{
	assert(instance->window == w);
	if(instance->input) instance->input->MouseButtonUsed(DetermineMouseButton(button),
														 DetermineAction(action));
}

void VisorGL::MouseScrolledGLFW(GLFWwindow* w, double x, double y)
{
	assert(instance->window == w);
	if(instance->input) instance->input->MouseScrolled(x, y);
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

GLenum VisorGL::PixelFormatToGL(PixelFormat f)
{
	static constexpr GLenum TypeList[] =
	{
		GL_R,
		GL_RG,
		GL_RGB,
		GL_RGBA,

		GL_R,
		GL_RG,
		GL_RGB,
		GL_RGBA,

		GL_R,
		GL_RG,
		GL_RGB,
		GL_RGBA,

		GL_R,
		GL_RG,
		GL_RGB,
		GL_RGBA
	};
	return TypeList[static_cast<int>(f)];
}

GLenum VisorGL::PixelFormatToSizedGL(PixelFormat f)
{
	static constexpr GLenum TypeList[] =
	{
		GL_R8,
		GL_RG8,
		GL_RGB8,
		GL_RGBA8,

		GL_R16,
		GL_RG16,
		GL_RGB16,
		GL_RGBA16,

		GL_R16F,
		GL_RG16F,
		GL_RGB16F,
		GL_RGBA16F,

		GL_R32F,
		GL_RG32F,
		GL_RGB32F,
		GL_RGBA32F
	};
	return TypeList[static_cast<int>(f)];
}

void VisorGL::ProcessCommand(const VisorGLCommand& c)
{
	switch(c.type)
	{
		case VisorGLCommand::RESET_IMAGE:
		{
			// Unbind and Delete
			glBindTexture(GL_TEXTURE_2D, 0);
			glDeleteTextures(1, &texture);
			glGenTextures(1, &texture);

			// Set Storage
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexStorage2D(GL_TEXTURE_2D, 1, PixelFormatToSizedGL(c.format),
						   c.startOrSize[0], c.startOrSize[1]);

			// Clear with black
			GLuint clearData[4] = {0, 0, 0, 0};
			glClearTexImage(GL_TEXTURE_2D, 1, GL_RGBA, GL_UNSIGNED_INT, clearData);

			texPixFormat = c.format;
			break;
		}
		case VisorGLCommand::SET_PORTION:
		{
			Vector2i size = c.end - c.startOrSize;

			glBindTexture(GL_TEXTURE_2D, texture);
			glTexSubImage2D(GL_TEXTURE_2D, 1,
							c.startOrSize[0], c.startOrSize[1],
							size[0], size[1],
							PixelFormatToGL(texPixFormat),
							GL_FLOAT,
							c.data.data());
			break;
		}
	};
}

void VisorGL::RenderImage()
{
	// Clear
	glViewport(0, 0, viewportSize[0], viewportSize[1]);
	glClear(GL_COLOR_BUFFER_BIT);
	// Draw
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

VisorGL::VisorGL(const VisorOptions& opts)
	: input(nullptr)
	, window(nullptr)
	, open(false)
{
	instance = this;

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

	if(opts.stereoOn)
		glfwWindowHint(GLFW_STEREO, GL_TRUE);

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
	
	// Set Callbacks
	glfwSetWindowPosCallback(window, VisorGL::WindowPosGLFW);
	glfwSetFramebufferSizeCallback(window, VisorGL::WindowFBGLFW);
	glfwSetWindowSizeCallback(window, VisorGL::WindowSizeGLFW);
	glfwSetWindowCloseCallback(window, VisorGL::WindowCloseGLFW);
	glfwSetWindowRefreshCallback(window, VisorGL::WindowRefreshGLFW);
	glfwSetWindowFocusCallback(window, VisorGL::WindowFocusedGLFW);
	glfwSetWindowIconifyCallback(window, VisorGL::WindowMinimizedGLFW);

	glfwSetKeyCallback(window, VisorGL::KeyboardUsedGLFW);
	glfwSetCursorPosCallback(window, VisorGL::MouseMovedGLFW);
	glfwSetMouseButtonCallback(window, VisorGL::MousePressedGLFW);
	glfwSetScrollCallback(window, VisorGL::MouseScrolledGLFW);

	// Shaders
	vertPP = ShaderGL(ShaderType::VERTEX, "PProcessGeneric.vert");
	fragPP = ShaderGL(ShaderType::VERTEX, "PProcessGeneric.frag");

	// Texture
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexStorage2D(GL_TEXTURE_2D, 1, PixelFormatToSizedGL(opts.iFormat),
				   opts.iSize[0], opts.iSize[1]);

	// Sampler
	glGenSamplers(1, &linearSampler);
	glSamplerParameteri(linearSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glSamplerParameteri(linearSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	// Buffer
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindVertexBuffer(0, vBuffer, 0, sizeof(float) * 2);
	glEnableVertexAttribArray(IN_POS);
	glVertexAttribFormat(IN_POS, 2, GL_FLOAT, false, 0);
	glVertexAttribBinding(IN_POS, 0);

	// Pre-Bind Everything
	// States
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	// FBO
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glColorMask(true, true, true, true);
	glDepthMask(false);
	glStencilMask(false);
	
	// Bind Shaders
	vertPP.Bind();
	fragPP.Bind();

	// Bind Texture & Sampler
	glActiveTexture(GL_TEXTURE0 + T_INPUT);
	glBindTexture(GL_TEXTURE_2D, texture);
	glBindSampler(T_INPUT, linearSampler);

	// Bind VAO
	glBindVertexArray(vao);


	// Finally Show Window
	glfwShowWindow(window);
	open = true;
}

VisorGL::~VisorGL()
{
	if(window != nullptr) glfwDestroyWindow(window);
	glfwTerminate();
}

//VisorGL& VisorGL::Instance()
//{
//	if(instance == nullptr)
//		instance = std::make_unique<VisorGL>();
//	return *instance;
//}

bool VisorGL::IsOpen()
{
	return open;
}

void VisorGL::Present()
{
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void VisorGL::Render()
{
	// Consume commands
	// TODO: optimize this skip multiple reset commands
	// just process the last and other commands afterwards
	while(!commandList.empty())
	{
		VisorGLCommand c;
		{
			std::unique_lock<std::mutex> lock(mutexCommand);
			c = std::move(commandList.back());
			commandList.pop_back();
		}
		ProcessCommand(c);
	}

	// Render Image
	RenderImage();
}

void VisorGL::SetInputScheme(VisorInputI* input)
{
	// TODO:
}

void VisorGL::ResetImageBuffer(const Vector2i& imageSize, PixelFormat f)
{
	VisorGLCommand command;
	command.type = VisorGLCommand::RESET_IMAGE;
	command.format = f;
	command.startOrSize = imageSize;
	{
		std::unique_lock<std::mutex> lock(mutexCommand);
		commandList.push_back(command);
	}
}

void VisorGL::SetImagePortion(const Vector2i& start,
							  const Vector2i& end,
							  const std::vector<float> data)
{

	VisorGLCommand command;
	command.type = VisorGLCommand::SET_PORTION;
	command.startOrSize = start;
	command.end = end;
	command.data = std::move(data);
	{
		std::unique_lock<std::mutex> lock(mutexCommand);
		commandList.push_back(command);
	}
}

void VisorGL::SetWindowSize(const Vector2i& size)
{
	glfwSetWindowSize(window, size[0], size[1]);
	viewportSize = size;
}

void VisorGL::SetFPSLimit(float f)
{
	int interval = static_cast<int>(1000.0f / f);
	glfwSwapInterval(interval);
}