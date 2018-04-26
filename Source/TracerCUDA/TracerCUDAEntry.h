#pragma once

#ifdef METU_SHARED_TRACERCUDA
#define METU_SHARED_TRACERCUDA_ENTRY_POINT __declspec(dllexport)
#else
#define METU_SHARED_TRACERCUDA_ENTRY_POINT __declspec(dllimport)
#endif

#include <memory>

class TracerI;

METU_SHARED_TRACERCUDA_ENTRY_POINT std::unique_ptr<TracerI> CreateTracerCUDA();