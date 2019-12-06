#pragma once

#include <filesystem>

namespace Utilitiy
{
    inline std::string MergeFileFolder(const std::string& folder,
                                       const std::string& file)
    {
        // Return file if file is absolute
        // If not make file relative to the path (concat)

        std::filesystem::path filePath(file);
        if(filePath.is_absolute())
            return file;
        else
        {
            std::filesystem::path folderPath(folder);
            auto mergedPath = folderPath / file;
            return mergedPath.string();
        }
    }
}