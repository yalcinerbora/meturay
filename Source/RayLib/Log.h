#pragma once

#include <string>

#include <fmt/core.h>
#include <fmt/color.h>

// Debug
#ifdef METU_DEBUG
    static constexpr bool IS_DEBUG_MODE = true;

    template<class... Args>
    inline void METU_DEBUG_LOG(Args&&... args)
    {
        std::string s = fmt::format(std::forward<Args>(args)...);
        fmt::print(stdout, "{:s}: {:s}\n",
                   fmt::format(fg(fmt::color::blanched_almond), "Debug"),
                   s);
    }
#else
    static constexpr bool IS_DEBUG_MODE = false;
    template<class... Args>
    inline constexpr void METU_DEBUG_LOG(Args&&...) {}
#endif

template<class... Args>
inline void METU_LOG(Args&&... args)
{
    std::string s = fmt::format(std::forward<Args>(args)...);
    fmt::print(stdout, "{:s}\n", s);
}

template<class... Args>
inline void METU_ERROR_LOG(Args&&... args)
{
    std::string s = fmt::format(std::forward<Args>(args)...);
    fmt::print(stderr, "{:s}: {:s}\n",
               fmt::format(fg(fmt::color::crimson), "Error"),
               s);
}