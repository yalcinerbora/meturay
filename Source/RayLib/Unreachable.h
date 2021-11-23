#pragma once

#ifdef _WIN32
    #define UNREACHABLE() //__assume(0)
#endif
#ifdef __linux__
    #define UNREACHABLE() __builtin_unreachable()
#endif