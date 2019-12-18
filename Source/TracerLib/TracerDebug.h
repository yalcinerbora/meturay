#pragma once

#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "RayLib/Log.h"
#include "RayLib/HitStructs.h"

#include "RayStructs.h"
#include "CudaConstants.h"

class ImageMemory;
struct DefaultLeaf;

namespace Debug
{
    void                PrintHitPairs(const RayId* ids, const HitKey* keys, size_t count);
    void                WriteHitPairs(const RayId* ids, const HitKey* keys, size_t count, const std::string& file);
    void                DumpImage(const std::string& fName,
                                  const ImageMemory&);

    // Memory Debugging
    template<class T>
    static void     DumpMemToFile(const std::string& fName,
                                  const T* mPtr, size_t count,
                                  const char* seperator = "\n",
                                  bool hex = false);
    template<class T>
    static void     DumpMemToStdout(const T* mPtr, size_t count,
                                    const char* seperator = "\n",
                                    bool hex = false);
    template<class T>
    static void     DumpMemToStream(std::ostream& s,
                                    const T* mPtr, size_t count,
                                    const char* seperator = "\n",
                                    bool hex = false);

    namespace Detail
    {
        template <class T>
        void OutputData(std::ostream& s, const T* hits, size_t count);
    }
}

// Some Print Func Definitions
extern std::ostream& operator<<(std::ostream& stream, const RayGMem&);
extern std::ostream& operator<<(std::ostream& stream, const HitKey&);
extern std::ostream& operator<<(std::ostream& stream, const Vector2ul&);
extern std::ostream& operator<<(std::ostream& stream, const Vector2f&);
extern std::ostream& operator<<(std::ostream& stream, const Vector4f&);
extern std::ostream& operator<<(std::ostream& stream, const DefaultLeaf&);

template<class T>
void Debug::DumpMemToFile(const std::string& fName,
                          const T* mPtr, size_t count,
                          const char* seperator, bool hex)
{
    std::ofstream file(fName);
    DumpMemToStream(file, mPtr, count, seperator, hex);
}

template<class T>
void Debug::DumpMemToStdout(const T* mPtr, size_t count,
                            const char* seperator, bool hex)
{
    DumpMemToStream(std::cout, mPtr, count, seperator, hex);
}

template<class T>
void Debug::DumpMemToStream(std::ostream& s,
                            const T* mPtr, size_t count,
                            const char* seperator, bool hex)
{
    CudaSystem::SyncAllGPUs();
    if(hex) s << std::hex;
    for(size_t i = 0; i < count; i++)
    {
        s << mPtr[i] << seperator;
    }
    if(hex) s << std::dec;
}