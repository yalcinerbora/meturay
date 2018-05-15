#include "VisorGLEntry.h"
#include "VisorGL.h"

METU_SHARED_VISORGL_ENTRY_POINT std::unique_ptr<VisorViewI> CreateVisorGL()
{
	return std::make_unique<VisorGL>();
}