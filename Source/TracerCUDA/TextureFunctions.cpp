#include "TextureFunctions.h"
#include <filesystem>

const TextureLoader::ExtToTypeMap TextureLoader::ExtToTypeList =
{
    {"bmp", FIF_BMP},
    {"dds", FIF_DDS},
    {"exr", FIF_EXR},
    {"hdr", FIF_HDR},

    {"jpg", FIF_JPEG},
    {"jpeg", FIF_JPEG},
    {"jif", FIF_JPEG},
    {"jpe", FIF_JPEG},

    {"png", FIF_PNG},
    {"ppm", FIF_PPM},

    {"tga", FIF_TARGA},
    {"targa", FIF_TARGA},

    {"tif", FIF_TIFF},
    {"tiff", FIF_TIFF}

    // TODO: add more if necessary
};

 SceneError TextureLoader::ConvertFileExtToFormat(FREE_IMAGE_FORMAT& f,
                                       const std::string& filePath)
{
    // Convert to actual path
    std::filesystem::path p(filePath);
    std::string ext = p.extension().string().substr(1);

    auto i = ExtToTypeList.cend();
    if((i = ExtToTypeList.find(ext)) != ExtToTypeList.cend())
        return SceneError::UNKNOWN_TEXTURE_TYPE;
    else
    {
        f = i->second;
        return SceneError::OK;
    }
}

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
    SceneError e = SceneError::OK;

    if((e = ConvertFileExtToFormat(f, filePath)) != SceneError::OK)
        return e;

    FIBITMAP* imgCPU = FreeImage_Load(f, filePath.c_str(), 0);

    if(imgCPU)
    {
        // Bit per pixel
        uint32_t bpp = FreeImage_GetBPP(imgCPU);


        FreeImage_Unload(imgCPU);
    }
    else return SceneError::UNABLE_TO_LOAD_TEXTURE;
}