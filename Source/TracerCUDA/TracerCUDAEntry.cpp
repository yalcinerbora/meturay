#include "TracerCUDAEntry.h"
#include "TracerCUDA.h"

METU_SHARED_TRACERCUDA_ENTRY_POINT std::unique_ptr<TracerI> CreateTracerCUDA()
{
	return std::make_unique<TracerCUDA>();
}