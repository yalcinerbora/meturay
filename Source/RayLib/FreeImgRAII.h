#pragma once

#include <FreeImage.h>

// Simple RAII Pattern for FreeImage Struct
// in order to prevent leaks
class FreeImgRAII
{
    private:
        FIBITMAP* imgCPU;
    protected:
    public:
        // Constructors & Destructor
        FreeImgRAII(FREE_IMAGE_FORMAT fmt, 
                    const char* fName, int flags = 0)
        {
            imgCPU = FreeImage_Load(fmt, fName, flags);
        }
        FreeImgRAII(const FreeImgRAII&) = delete;
        FreeImgRAII& operator=(const FreeImgRAII&) = delete;
        ~FreeImgRAII() { if(imgCPU) FreeImage_Unload(imgCPU); }
        //
        operator FIBITMAP*() { return imgCPU; }
        operator const FIBITMAP* () const { return imgCPU; }
};