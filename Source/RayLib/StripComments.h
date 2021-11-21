#pragma once

#include <istream>
#include <sstream>
#include <regex>

namespace Utility
{
    std::stringstream StripComments(std::istream& source);
}

inline std::stringstream Utility::StripComments(std::istream& source)
{
    // Use Regex to Strip Comments
    std::regex commentre("/\\*([\\s\\S]*?)\\*/");
    std::regex commentli("//(.*)[\\r\\n|\\n]");

    std::string all(std::istreambuf_iterator<char>(source), {});
    std::string modified = std::regex_replace(all, commentre, "");
    modified = std::regex_replace(modified, commentli, "\n");

    std::stringstream result(modified);
    return result;
}