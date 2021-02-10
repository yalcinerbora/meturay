#include "TextureFunctions.h"
#include <filesystem>

TextureLoader::TextureLoader()
{
    FreeImage_Initialise();
}

TextureLoader::~TextureLoader()
{
    FreeImage_DeInitialise();
}

SceneError TextureLoader::LoadTexture(const std::string& filePath)
{
    FREE_IMAGE_FORMAT f;
    // Check file to find type
    if((f = FreeImage_GetFileType(filePath.c_str())) == FIF_UNKNOWN)
        // Use file extension to determine type
        f = FreeImage_GetFIFFromFilename(filePath.c_str());
    // If it is still unknown just return error
    if(f == FIF_UNKNOWN)
        return SceneError::UNKNOWN_TEXTURE_TYPE;

    FIBITMAP* imgCPU = FreeImage_Load(f, filePath.c_str(), 0);
    if(imgCPU)
    {
        // Bit per pixel
        uint32_t bpp = FreeImage_GetBPP(imgCPU);




        FreeImage_Unload(imgCPU);
    }
    else return SceneError::UNABLE_TO_LOAD_TEXTURE;


    return SceneError::OK;
}