#pragma once

#include <string>

namespace Utility
{
    std::string MergeFileFolder(const std::string& folder,
                                const std::string& file);

    std::string PrependToFileInPath(const std::string& path,
                                    const std::string& prefix);

    void ForceMakeDirectoriesInPath(const std::string& path);
}