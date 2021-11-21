
#include "EntryPoint.h"
#include "ImageIO.h"

extern "C" METU_SHARED_IMAGEIO_ENTRY_POINT
ImageIOI* ImageIOInstance()
{
    static std::unique_ptr<ImageIO> instance = nullptr;

    if(instance == nullptr)
        instance = std::make_unique<ImageIO>();
    return instance.get();

    /*return ImageIO::Instance();*/
}