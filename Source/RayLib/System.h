#pragma once

#define ERROR_CHECK(ErrType, e) \
if(e != ErrType::OK) \
{\
    METU_ERROR_LOG(static_cast<std::string>(e)); \
    return false;\
}

#define ERROR_CHECK_INT(ErrType, e) \
if(e != ErrType::OK) \
{\
    METU_ERROR_LOG(static_cast<std::string>(e)); \
    return 1;\
}

#ifdef _WIN32
    #define METURAY_WIN
    #define MRAY_DLL_IMPORT __declspec(dllimport)
    #define MRAY_DLL_EXPORT __declspec(dllexport)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    static inline bool EnableVTMode()
    {
        // Set output mode to handle virtual terminal sequences
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if(hOut == INVALID_HANDLE_VALUE)
        {
            return false;
        }

        DWORD dwMode = 0;
        if(!GetConsoleMode(hOut, &dwMode))
        {
            return false;
        }

        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        if(!SetConsoleMode(hOut, dwMode))
        {
            return false;
        }
        return true;
    }


#endif

#ifdef __linux__
    #define METURAY_LINUX
    //#define MRAY_DLL_IMPORT __attribute__((dllimport))
    //#define MRAY_DLL_EXPORT __attribute__((dllexport))
    #define MRAY_DLL_IMPORT
    #define MRAY_DLL_EXPORT
    #include <unistd.h>
    static inline bool EnableVTMode()
    {
        return true;
    }
#endif
