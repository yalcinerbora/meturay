#include "VisorGLEntry.h"
#include "VisorGL.h"

extern "C" METU_SHARED_VISORGL_ENTRY_POINT
VisorI* CreateVisorGL(const VisorOptions& opts,
                                const Vector2i& imgRes,
                                const PixelFormat& f)
{
    return new VisorGL(opts, imgRes, f);
}

extern "C" METU_SHARED_VISORGL_ENTRY_POINT
void DeleteVisorGL(VisorI* v)
{
    if(v) delete v;
}