#pragma once

#include <string>
#include <cstring>

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

    inline std::string CopyU8ToString(const std::u8string& u8S)
    {
        static_assert(sizeof(char8_t) == sizeof(char), "char8_t char size mismatch");
        std::string string;
        string.resize(u8S.size(), u8'\0');
        std::memcpy(string.data(), u8S.data(), u8S.size() * sizeof(char));
        return string;
    }
}
