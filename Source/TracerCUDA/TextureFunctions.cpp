#include "TextureFunctions.h"

TextureLoader::TextureLoader()
{
    FreeImage_Initialise();
}

TextureLoader::~TextureLoader()
{
    FreeImage_DeInitialise();
}