#include "ImageIO.h"
#include "FreeImage.h"

bool ImageIO::WriteAsPNG(const std::vector<Vector3>& image,
						 const Vector2ui& size,
						 const std::string& fileName)
{
	FreeImage_Initialise();
	auto* bitmap = FreeImage_Allocate(size[0], size[1], 24);

	for(uint32_t j = 0; j < size[1]; j++)
	for(uint32_t i = 0; i < size[0]; i++)
	{
		uint32_t jImg = size[0] - j - 1;

		RGBQUAD color;
		Vector3 rgbImage = image[jImg * size[0] + i];

		rgbImage.ClampSelf(0.0f, 1.0f);
		rgbImage *= 255.0f;

		color.rgbRed = static_cast<BYTE>(rgbImage[0]);
		color.rgbGreen = static_cast<BYTE>(rgbImage[1]);
		color.rgbBlue = static_cast<BYTE>(rgbImage[2]);

		FreeImage_SetPixelColor(bitmap, i , j, &color);
	}
	bool result = FreeImage_Save(FIF_PNG, bitmap, fileName.c_str());
	FreeImage_Unload(bitmap);
	FreeImage_DeInitialise();
	return result;
}