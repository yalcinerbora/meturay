#pragma once

#include "RayLib/VisorI.h"
#include "RayLib/System.h"

#ifdef METU_SHARED_VISORGL
#define METU_SHARED_VISORGL_ENTRY_POINT MRAY_DLL_EXPORT
#else
#define METU_SHARED_VISORGL_ENTRY_POINT MRAY_DLL_IMPORT
#endif

extern "C" METU_SHARED_VISORGL_ENTRY_POINT
VisorI* CreateVisorGL(const VisorOptions&,
                      const Vector2i& imgRes,
                      const PixelFormat&);

extern "C" METU_SHARED_VISORGL_ENTRY_POINT
VisorI * CreateGuideDebugGL(const Vector2i& winSize,
                            const std::u8string& guideDebugFile);


extern "C" METU_SHARED_VISORGL_ENTRY_POINT
void DeleteVisorGL(VisorI*);