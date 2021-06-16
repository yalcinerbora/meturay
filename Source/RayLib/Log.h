#pragma once

#include <string>

// Debug
#ifdef METU_DEBUG
    static constexpr bool IS_DEBUG_MODE = true;

    template<class... Args>
    inline void METU_DEBUG_LOG(const char* string, Args... args)
    {
        std::string s;
        s += "\33[2K\r";
        s += string;
        s += "\n";
        fprintf(stdout, s.c_str(), args...);
    }
#else
    static constexpr bool IS_DEBUG_MODE = false;
    template<class... Args>
    inline constexpr void METU_DEBUG_LOG(const char* string, Args... args) {}
#endif

template<class... Args>
inline void METU_LOG(const char* string, Args... args)
{
    std::string s;
    //s += "\33[2K\r";
    s += string;
    s += "\n";
    fprintf(stdout, s.c_str(), args...);
}

template<class... Args>
inline void METU_ERROR_LOG(const char* string, Args... args)
{
    std::string s;
    //s += "\33[2K\r";
    s += "Error: ";
    s += string;
    s += "\n";
    fprintf(stderr, s.c_str(), args...);
}