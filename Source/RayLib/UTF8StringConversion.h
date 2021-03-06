#pragma once

#include <string>

namespace Utility
{
    // Converts string to u8
    inline std::u8string CopyStringU8(const std::string& s)
    {
        static_assert(sizeof(char8_t) == sizeof(char), "char8_t char size mismatch");
        std::u8string u8String;
        u8String.resize(s.size(), u8'\0');
        std::memcpy(u8String.data(), s.data(), s.size() * sizeof(char));
        return u8String;
    }
}
