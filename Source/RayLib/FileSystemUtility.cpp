#include "FileSystemUtility.h"

#include <filesystem>

std::string Utility::MergeFileFolder(const std::string& folder,
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

std::string Utility::PrependToFileInPath(const std::string& path,
                                         const std::string& prefix)
{
    std::filesystem::path filePath(path);
    std::filesystem::path result = (filePath.parent_path() /
                                    (prefix + filePath.filename().string()));
    return result.string();
}

void Utility::ForceMakeDirectoriesInPath(const std::string& path)
{
    std::filesystem::path filePath(path);
    if(std::filesystem::is_directory(filePath))
        std::filesystem::create_directories(filePath);
    else
        std::filesystem::create_directories(filePath.parent_path());
}