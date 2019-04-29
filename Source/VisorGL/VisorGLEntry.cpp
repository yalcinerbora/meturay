#include "VisorGLEntry.h"
#include "VisorGL.h"

METU_SHARED_VISORGL_ENTRY_POINT std::unique_ptr<VisorI> CreateVisorGL(const VisorOptions& opts)
{
    return std::make_unique<VisorGL>(opts);
}