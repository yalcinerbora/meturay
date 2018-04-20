#include "TracerCUDAEntry.h"

METU_SHARED_TRACERCUDA_ENTRY_POINT std::unique_ptr<TracerCUDA> CreateTracerCUDA()
{
	return std::make_unique<TracerCUDA>();
}