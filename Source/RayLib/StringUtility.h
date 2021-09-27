#pragma once

#include <string>

namespace Utility
{
    std::size_t ReplaceAll(std::string& inout, std::string_view what, std::string_view with);
}

// From 
// https://en.cppreference.com/w/cpp/string/basic_string/replace
inline std::size_t Utility::ReplaceAll(std::string& inout, std::string_view what, std::string_view with)
{
    std::size_t count = 0;
    for(std::string::size_type pos = 0;
        inout.npos != (pos = inout.find(what.data(), pos, what.length()));
        pos += with.length(), ++count)
    {
        inout.replace(pos, what.length(), with.data(), with.length());
    }
    return count;
}
