#include "VisorGLEntry.h"
#include "VisorGL.h"

METU_SHARED_VISORGL_ENTRY_POINT 
VisorI* __stdcall CreateVisorGL(const VisorOptions& opts)
{
    return new VisorGL(opts);
}

METU_SHARED_VISORGL_ENTRY_POINT
void __stdcall DeleteVisorGL(VisorI* v)
{
    if(v) delete v;
}