#pragma once

#ifdef METU_SHARED_VISORGL
#define METU_SHARED_VISORGL_ENTRY_POINT __declspec(dllexport)
#else
#define METU_SHARED_VISORGL_ENTRY_POINT __declspec(dllimport)
#endif

#include "RayLib/VisorI.h"

extern "C" METU_SHARED_VISORGL_ENTRY_POINT 
VisorI* __stdcall CreateVisorGL(const VisorOptions&,
                                const Vector2i& imgRes,
                                const PixelFormat&);

extern "C" METU_SHARED_VISORGL_ENTRY_POINT
void __stdcall DeleteVisorGL(VisorI*);