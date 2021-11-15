#pragma once

#include "RayLib/Log.h"
#include <cassert>

#ifdef METU_CUDA
#ifdef MRAY_OPTIX
    #include <optix_host.h>

    inline static constexpr void OptiXAssert(OptixResult code, const char *file, int line)
    {
        if(code != OPTIX_SUCCESS)
        {
            METU_ERROR_LOG("Optix Failure: {:s} {:s} {:d}", optixGetErrorString(code), file, line);
            assert(false);
        }
    }

    #ifdef METU_DEBUG
        #define OPTIX_CHECK(func) OptiXAssert((func), __FILE__, __LINE__)
        #define OPTIX_CHECK_ERROR(err) OptiXAssert(err, __FILE__, __LINE__)
    #else
        #define OPTIX_CHECK_ERROR(err)
        #define OPTIX_CHECK(func) func
    #endif
#endif
#endif
