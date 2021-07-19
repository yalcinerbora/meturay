#pragma once

#include <fstream>
#include <vector>

namespace Utility
{
    template <class T>
    void DumpStdVectorToFile(const std::vector<T>&, const std::string& filePath,
                             bool append = false);
}

template <class T>
void Utility::DumpStdVectorToFile(const std::vector<T>& data, const std::string& filePath,
                                  bool append)
{
    std::ios::openmode mode = std::ios::binary;        
    if(append) mode |= std::ios::app;
    // Actual write operation
    std::ofstream file(filePath, std::ios::binary );
    file.write(reinterpret_cast<const char*>(data.data()),
               data.size() * sizeof(T));
}