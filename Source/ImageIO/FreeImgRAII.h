#pragma once

#include <FreeImage.h>

// Simple RAII Pattern for FreeImage Struct
// in order to prevent leaks
class FreeImgRAII
{
    private:
        FIBITMAP*       imgCPU;
    protected:
    public:
        // Constructors & Destructor
                        FreeImgRAII();
                        FreeImgRAII(FIBITMAP*);
                        FreeImgRAII(FREE_IMAGE_FORMAT fmt,
                                    const char* fName, int flags = 0);    
                        FreeImgRAII(const FreeImgRAII&) = delete;
                        FreeImgRAII(FreeImgRAII&&);
        FreeImgRAII&    operator=(const FreeImgRAII&) = delete;
        FreeImgRAII&    operator=(FreeImgRAII&&);
                        ~FreeImgRAII();
        // Type Cast Operators
        operator        FIBITMAP*();
        operator        const FIBITMAP*() const;

        Byte*           Data();
        const Byte*     Data() const;
        
        size_t          Pitch() const;
};

inline FreeImgRAII::FreeImgRAII()
    : imgCPU(nullptr)
{}

inline FreeImgRAII::FreeImgRAII(FIBITMAP* fbm)
    : imgCPU(fbm)
{}

inline FreeImgRAII::FreeImgRAII(FREE_IMAGE_FORMAT fmt,
                                const char* fName, int flags)
{
    imgCPU = FreeImage_Load(fmt, fName, flags);
}

inline FreeImgRAII::FreeImgRAII(FreeImgRAII&& other)
    : imgCPU(other.imgCPU)
{
    other.imgCPU = nullptr;
}

inline FreeImgRAII& FreeImgRAII::operator=(FreeImgRAII&& other)
{
    assert(&other != this);
    if(imgCPU) FreeImage_Unload(imgCPU);
    imgCPU = other.imgCPU;
    other.imgCPU = nullptr;
    return *this;
}

inline FreeImgRAII::~FreeImgRAII() 
{ 
    if(imgCPU) FreeImage_Unload(imgCPU);
}

inline FreeImgRAII::operator FIBITMAP* () { return imgCPU; }
inline FreeImgRAII::operator const FIBITMAP* () const { return imgCPU; }

inline Byte* FreeImgRAII::Data()
{
    return FreeImage_GetBits(imgCPU);
}

inline const Byte* FreeImgRAII::Data() const
{
    return FreeImage_GetBits(imgCPU);
}

inline size_t FreeImgRAII::Pitch() const
{
    return FreeImage_GetPitch(imgCPU);
}
