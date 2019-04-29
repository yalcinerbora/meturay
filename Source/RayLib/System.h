#pragma once

#ifdef _WIN32
    #define METURAY_WIN
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
#else
    static inline bool EnableVTMode()
    {}
#endif


#ifdef __linux__
    #define METURAY_UNIX
#else

#endif


