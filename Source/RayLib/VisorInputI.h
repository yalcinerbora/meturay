#pragma once
/**

MVisorInput Interface

Can be attached to a Visor to capture window actions


*/

#include <functional>
#include <map>

class VisorCallbacksI;

enum class KeyboardKeyType
{
	SPACE,
	APOSTROPHE,
	COMMA,
	MINUS,
	PERIOD,
	SLASH,
	NUMBER_0,
	NUMBER_1,
	NUMBER_2,
	NUMBER_3,
	NUMBER_4,
	NUMBER_5,
	NUMBER_6,
	NUMBER_7,
	NUMBER_8,
	NUMBER_9,
	SEMICOLON,
	EQUAL,
	A,
	B,
	C,
	D,
	E,
	F,
	G,
	H,
	I,
	J,
	K,
	L,
	M,
	N,
	O,
	P,
	Q,
	R,
	S,
	T,
	U,
	V,
	W,
	X,
	Y,
	Z,
	LEFT_BRACKET,
	BACKSLASH,
	RIGHT_BRACKET,
	GRAVE_ACCENT,
	WORLD_1,
	WORLD_2,
	ESCAPE,
	ENTER,
	TAB,
	BACKSPACE,
	INSERT,
	DELETE_KEY,
	RIGHT,
	LEFT,
	DOWN,
	UP,
	PAGE_UP,
	PAGE_DOWN,
	HOME,
	END,
	CAPS_LOCK,
	SCROLL_LOCK,
	NUM_LOCK,
	PRINT_SCREEN,
	PAUSE,
	F1,
	F2,
	F3,
	F4,
	F5,
	F6,
	F7,
	F8,
	F9,
	F10,
	F11,
	F12,
	F13,
	F14,
	F15,
	F16,
	F17,
	F18,
	F19,
	F20,
	F21,
	F22,
	F23,
	F24,
	F25,
	KP_0,
	KP_1,
	KP_2,
	KP_3,
	KP_4,
	KP_5,
	KP_6,
	KP_7,
	KP_8,
	KP_9,
	KP_DECIMAL,
	KP_DIVIDE,
	KP_MULTIPLY,
	KP_SUBTRACT,
	KP_ADD,
	KP_ENTER,
	KP_EQUAL,
	LEFT_SHIFT,
	LEFT_CONTROL,
	LEFT_ALT,
	LEFT_SUPER,
	RIGHT_SHIFT,
	RIGHT_CONTROL,
	RIGHT_ALT,
	RIGHT_SUPER,
	MENU
};

enum class MouseButtonType
{
	LEFT,
	RIGHT,
	MIDDLE,
	BUTTON_4,
	BUTTON_5,
	BUTTON_6,
	BUTTON_7,
	BUTTON_8
};

enum class KeyAction
{
	PRESSED,
	RELEASED,
	REPEATED,
};

using KeyCallbacks = std::multimap<std::pair<KeyboardKeyType, KeyAction>, std::function<void()>>;
using MouseButtonCallbacks = std::multimap<std::pair<MouseButtonType, KeyAction>, std::function<void()>>;

class VisorInputI
{
	private:
		KeyCallbacks						keyCallbacks;
		MouseButtonCallbacks				buttonCallbacks;

		void								KeyboardUsedWithCallbacks(KeyboardKeyType key, KeyAction action);
		void								MouseButtonUsedWithCallbacks(MouseButtonType button, KeyAction action);

	protected:
	public:
		virtual								~VisorInputI() = default;

		// Interface
		virtual void						AttachVisorCallback(VisorCallbacksI&) = 0;

		virtual void						WindowPosChanged(int posX, int posY) = 0;
		virtual void						WindowFBChanged(int fbWidth, int fbHeight) = 0;
		virtual void						WindowSizeChanged(int width, int height) = 0;
		virtual void						WindowClosed() = 0;
		virtual void						WindowRefreshed() = 0;
		virtual void						WindowFocused(bool) = 0;
		virtual void						WindowMinimized(bool) = 0;

		virtual void						MouseScrolled(double xOffset, double yOffset) = 0;
		virtual void						MouseMoved(double x, double y) = 0;

		virtual void						KeyboardUsed(KeyboardKeyType key, KeyAction action) = 0;
		virtual void						MouseButtonUsed(MouseButtonType button, KeyAction action) = 0;

		// Defining Custom Callback
		template <class Function, class... Args>
		void								AddKeyCallback(KeyboardKeyType, KeyAction,
														   Function&& f, Args&&... args);
		template <class Function, class... Args>
		void								AddButtonCallback(MouseButtonType, KeyAction,
															  Function&& f, Args&&... args);
};

template <class Function, class... Args>
void VisorInputI::AddKeyCallback(KeyboardKeyType key, KeyAction action,
								 Function&& f, Args&&... args)
{
	std::function<void()> func = std::bind(f, args...);
	keyCallbacks.emplace(std::make_pair(key, action), func);
}

template <class Function, class... Args>
void VisorInputI::AddButtonCallback(MouseButtonType button, KeyAction action,
									Function&& f, Args&&... args)
{
	std::function<void()> func = std::bind(f, args...);
	buttonCallbacks.emplace(std::make_pair(button, action), func);
}

inline void VisorInputI::KeyboardUsedWithCallbacks(KeyboardKeyType key, KeyAction action)
{
	KeyboardUsed(key, action);

	auto range = keyCallbacks.equal_range(std::make_pair(key, action));
	for(auto it = range.first; it != range.second; ++it)
	{
		// Call Those Functions
		it->second();
	}
}

inline void VisorInputI::MouseButtonUsedWithCallbacks(MouseButtonType button, KeyAction action)
{
	MouseButtonUsed(button, action);

	auto range = buttonCallbacks.equal_range(std::make_pair(button, action));
	for(auto it = range.first; it != range.second; ++it)
	{
		// Call Those Functions
		it->second();
	}
}