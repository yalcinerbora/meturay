#pragma once

#include "ImageIOI.h"
#include "RayLib/System.h"

#ifdef METU_SHARED_IMAGE_IO
    #define METU_SHARED_IMAGEIO_ENTRY_POINT MRAY_DLL_EXPORT
#else
    #define METU_SHARED_IMAGEIO_ENTRY_POINT MRAY_DLL_IMPORT
#endif

extern "C" METU_SHARED_IMAGEIO_ENTRY_POINT
ImageIOI* ImageIOInstance();