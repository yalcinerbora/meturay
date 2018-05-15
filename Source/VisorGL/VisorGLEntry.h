#pragma once

#ifdef METU_SHARED_VISORGL
#define METU_SHARED_VISORGL_ENTRY_POINT __declspec(dllexport)
#else
#define METU_SHARED_VISORGL_ENTRY_POINT __declspec(dllimport)
#endif

#include <memory>
#include "RayLib/VisorI.h"

METU_SHARED_VISORGL_ENTRY_POINT std::unique_ptr<VisorViewI> CreateVisorGL();