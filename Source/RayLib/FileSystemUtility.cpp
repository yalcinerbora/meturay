#include "FileSystemUtility.h"
#include "System.h"

#include <filesystem>
#include <cassert>

std::string Utility::PathFile(const std::string& path)
{
    std::filesystem::path filePath(path);
    std::filesystem::path file(filePath.filename());
    return file.string();
}

std::string Utility::PathFolder(const std::string& path)
{
    std::filesystem::path filePath(path);
    std::filesystem::path parent(filePath.parent_path());
    return parent.string();
}

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

#include "Log.h"

std::vector<std::string> Utility::ListFilesInFolder(const std::string& folder,
                                                    const std::regex& regex)
{
    std::filesystem::path folderPath(folder);
    std::vector<std::string> result;
    // List directories
    for(auto const& file : std::filesystem::directory_iterator{folderPath})
    {
        if(std::regex_match(file.path().filename().string(), regex))
            result.push_back(file.path().string());
    }
    return result;
}

bool Utility::CheckFileExistance(const std::string& path)
{
    return std::filesystem::exists(path);
}

std::string Utility::CurrentExecPath()
{
    static constexpr size_t MAX_PATH_LENGTH = 512;

    #ifdef METURAY_WIN
        std::wstring result(MAX_PATH_LENGTH, '\0');
        GetModuleFileName(NULL, result.data(),
                          static_cast<DWORD>(result.size()));
        return std::filesystem::path(result).parent_path().string();
    #endif
    #ifdef METURAY_LINUX
        char result[MAX_PATH_LENGTH + 1];
        int error = readlink("/proc/self/exe", result, MAX_PATH_LENGTH);
        assert(error != -1);
        result[error + 1] = '\0';

        std::string resultStr(result);
        return Utility::PathFolder(resultStr);
    #endif
}