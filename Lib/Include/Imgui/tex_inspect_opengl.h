// ImGuiTexInspect, a texture inspector widget for dear imgui
#pragma once

#define IMGUI_IMPL_OPENGL_LOADER_GLEW

#include <stddef.h>
namespace ImGuiTexInspect
{
bool ImplOpenGL3_Init(const char *glsl_version = NULL);
void ImplOpenGl3_Shutdown();
} // namespace ImGuiTexInspect
