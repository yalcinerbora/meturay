#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "RayLib/Log.h"
#include "RayLib/HitStructs.h"
#include "RayLib/AABB.h"

#include "RayStructs.h"
#include "CudaSystem.h"

class ImageMemory;
struct DefaultLeaf;

struct STreeNode;
struct STreeGPU;
struct DTreeNode;
struct DTreeGPU;
struct PathGuidingNode;

namespace Debug
{
    void                PrintHitPairs(const RayId* ids, const HitKey* keys, size_t count);
    void                WriteHitPairs(const RayId* ids, const HitKey* keys, size_t count, const std::string& file);
    void                DumpImage(const std::string& fName,
                                  const ImageMemory&);
    void                DumpImage(const std::string& fName,
                                  const Vector4* iMem,
                                  const Vector2ui& resolution);
    void                DumpBitmap(const std::string& fName,
                                  const Byte* bits,
                                  const Vector2ui& resolution);

    // Memory Debugging
    template<class T>
    static void     DumpMemToFile(const std::string& fName,
                                  const T* mPtr, size_t count,
                                  bool append = false,
                                  bool hex = false,
                                  const char* seperator = "\n");

    template<class T>
    static void     DumpBatchedMemToFile(const std::string& fName,
                                         const T* mPtr,
                                         size_t batchCount,
                                         size_t totalCount,
                                         bool append = false,
                                         bool hex = false,
                                         const char* batchSeperator = "=============================",
                                         const char* elementSeperator = "\n");

    template<class T>
    static void     DumpMemToStdout(const T* mPtr, size_t count,
                                    bool hex = false,
                                    const char* seperator = "\n");
    template<class T>
    static void     DumpMemToStream(std::ostream& s,
                                    const T* mPtr, size_t count,
                                    bool hex = false,
                                    const char* seperator = "\n");
    template<class T>
    void            DumpBatchedMemToStream(std::ostream& s,
                                           const T* mPtr,
                                           size_t batchCount,
                                           size_t totalCount,
                                           bool hex,
                                           const char* batchSeperator,
                                           const char* elementSeperator);

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
extern std::ostream& operator<<(std::ostream& stream, const Vector3f&);
extern std::ostream& operator<<(std::ostream& stream, const Vector4f&);
extern std::ostream& operator<<(std::ostream& stream, const AABB3f&);
extern std::ostream& operator<<(std::ostream& stream, const DefaultLeaf&);
extern std::ostream& operator<<(std::ostream& stream, const STreeGPU&);
extern std::ostream& operator<<(std::ostream& stream, const STreeNode&);
extern std::ostream& operator<<(std::ostream& stream, const DTreeGPU&);
extern std::ostream& operator<<(std::ostream& stream, const DTreeNode&);
extern std::ostream& operator<<(std::ostream& stream, const PathGuidingNode&);

template<class T>
void Debug::DumpMemToFile(const std::string& fName,
                          const T* mPtr, size_t count,
                          bool append, bool hex,
                          const char* seperator)
{
    std::ofstream file;
    if(append)
    {
        file = std::ofstream(fName, std::ofstream::app);
        file << "============================" << std::endl;
    }
    else
        file = std::ofstream(fName);
    DumpMemToStream(file, mPtr, count, seperator, hex);
}

template<class T>
void Debug::DumpBatchedMemToFile(const std::string& fName,
                                 const T* mPtr,
                                 size_t batchCount,
                                 size_t totalCount,
                                 bool append, bool hex,
                                 const char* batchSeperator,
                                 const char* elementSeperator)
{
    std::ofstream file;
    if(append)
    {
        file = std::ofstream(fName, std::ofstream::app);
        file << "============================" << std::endl;
    }
    else
        file = std::ofstream(fName);
    DumpBatchedMemToStream(file, mPtr,
                           batchCount, totalCount,
                           hex,
                           batchSeperator,
                           elementSeperator);
}

template<class T>
void Debug::DumpMemToStdout(const T* mPtr, size_t count,
                            bool hex, const char* seperator)
{
    DumpMemToStream(std::cout, mPtr, count, hex, seperator);
}

template<class T>
void Debug::DumpMemToStream(std::ostream& s,
                            const T* mPtr, size_t count,
                            bool hex, const char* seperator)
{
    CUDA_CHECK(cudaDeviceSynchronize());

    if(hex) s << std::hex;
    for(size_t i = 0; i < count; i++)
    {
        s << mPtr[i] << seperator;
    }
    if(hex) s << std::dec;
}

template<class T>
void Debug::DumpBatchedMemToStream(std::ostream& s,
                                   const T* mPtr,
                                   size_t batchCount,
                                   size_t totalCount,
                                   bool hex,
                                   const char* batchSeperator,
                                   const char* elementSeperator)
{
    CUDA_CHECK(cudaDeviceSynchronize());

    if(hex) s << std::hex;
    for(size_t i = 0; i < totalCount; i++)
    {
        if(i != 0 && i % batchCount == 0)
            s << batchSeperator << "\n";

        s << mPtr[i] << elementSeperator;

    }
    if(hex) s << std::dec;
}