#pragma once

#include <string>
#include <vector>
#include <regex>

// These Utility is provided only because NVCC complains
// when it includes std::filesystem library
// nvcc does not use latest libstdc++ i guess but w/e

namespace Utility
{
    std::string                 PathFile(const std::string& path);
    std::string                 PathFolder(const std::string& path);

    std::string                 MergeFileFolder(const std::string& folder,
                                                const std::string& file);

    std::string                 PrependToFileInPath(const std::string& path,
                                                    const std::string& prefix);

    std::vector<std::string>    ListFilesInFolder(const std::string& folder,
                                                  const std::regex& regex = std::regex("*."));

    void                        ForceMakeDirectoriesInPath(const std::string& path);

    bool                        CheckFileExistance(const std::string& path);

    std::string                 CurrentExecPath();
}