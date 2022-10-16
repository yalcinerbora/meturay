#pragma once

#include <fstream>
#include <vector>
#include <filesystem>

namespace Utility
{
    template <class T>
    void DumpStdVectorToFile(const std::vector<T>&, const std::string& filePath,
                             bool append = false);

    template <class T>
    void DevourFileToStdVector(std::vector<T>&,
                               const std::string& filePath);
}

template <class T>
void Utility::DumpStdVectorToFile(const std::vector<T>& data, const std::string& filePath,
                                  bool append)
{
    std::ios::openmode mode = std::ios::binary;
    if(append) mode |= std::ios::app;
    // Actual write operation
    std::ofstream file(filePath, mode);
    file.write(reinterpret_cast<const char*>(data.data()),
               data.size() * sizeof(T));
}

template <class T>
void Utility::DevourFileToStdVector(std::vector<T>& data,
                                    const std::string& filePath)
{
    size_t size = std::filesystem::file_size(filePath);
    assert(size % sizeof(T) == 0);
    // Still allocate enough here "although assert will catch it"
    data.resize((size + sizeof(T) - 1) / sizeof(T));
    std::ifstream file(filePath, std::ios::binary);
    file.read(reinterpret_cast<char*>(data.data()), size);
}