#include "VisorGLEntry.h"

METU_SHARED_VISORGL_ENTRY_POINT std::unique_ptr<VisorGL> CreateVisorGL()
{
	return std::make_unique<VisorGL>();
}