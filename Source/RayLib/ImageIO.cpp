#include "ImageIO.h"
#include "FreeImage.h"

std::unique_ptr<ImageIO> ImageIO::instance = nullptr;

inline ImageIO::ImageIO()
{
	FreeImage_Initialise();
}

ImageIO::~ImageIO()
{
	FreeImage_DeInitialise();
}

ImageIO& ImageIO::System()
{
	if(instance == nullptr)
		instance = std::make_unique<ImageIO>();

	return *instance;
}

bool ImageIO::ReadHDR(std::vector<Vector4>& image,
					  Vector2ui& size,
					  const std::string& fileName) const
{
	FIBITMAP *dib2 = nullptr;
	FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(fileName.c_str());

	dib2 = FreeImage_Load(fif, fileName.c_str());
	if(dib2 == nullptr)
	{
		return false;
	}
	FIBITMAP *dib1 = FreeImage_ConvertToRGBAF(dib2);
	//FIBITMAP *dib1 = FreeImage_TmoReinhard05Ex(dib2);
	FreeImage_Unload(dib2);

	FREE_IMAGE_TYPE type = FreeImage_GetImageType(dib1);
	BITMAPINFOHEADER* header = FreeImage_GetInfoHeader(dib1);

	// Size
	size[0] = header->biWidth;
	size[1] = header->biHeight;
	image.resize(size[0] * size[1]);

	FREE_IMAGE_TYPE image_type = FreeImage_GetImageType(dib1);
	for(int j = 0; j < header->biHeight; j++)
	{		
		FIRGBAF *bits = (FIRGBAF *)FreeImage_GetScanLine(dib1, j);;
		for(int i = 0; i < header->biWidth; i++)
		{				
		/*	RGBQUAD rgb;
			bool fetched = FreeImage_GetPixelColor(dib1, i, header->biHeight - j - 1, &rgb);*/

			Vector4 pixel;
			//pixel[0] = static_cast<float>(rgb.rgbRed) / 255.0f;
			//pixel[1] = static_cast<float>(rgb.rgbGreen) / 255.0f;
			//pixel[2] = static_cast<float>(rgb.rgbBlue) / 255.0f;
			//pixel[3] = 0.0f;

			pixel[0] = bits[i].red * 2.5f;
			pixel[1] = bits[i].green * 2.5f;
			pixel[2] = bits[i].blue * 2.5f;
			pixel[3] = 0.0f;

			//if(pixel[0] > 1.0f)
			//	printf("%f ", pixel[0]);
			//if(pixel[1] > 1.0f)
			//	printf("%f ", pixel[1]);
			//if(pixel[2] > 1.0f)
			//	printf("%f ", pixel[2]);
			//printf("\n");

			image[j * header->biWidth + i] = pixel;
		}
	}
	return true;
}

bool ImageIO::WriteAsPNG(const Vector4f* image,
						 const Vector2ui& size,
						 const std::string& fileName) const
{
	auto* bitmap = FreeImage_Allocate(size[0], size[1], 24);

	for(uint32_t j = 0; j < size[1]; j++)
	for(uint32_t i = 0; i < size[0]; i++)
	{
		uint32_t jImg = size[1] - j - 1;

		RGBQUAD color;
		Vector4f rgbImage = image[jImg * size[0] + i];

		rgbImage.ClampSelf(0.0f, 1.0f);
		rgbImage *= 255.0f;

		color.rgbRed = static_cast<BYTE>(rgbImage[0]);
		color.rgbGreen = static_cast<BYTE>(rgbImage[1]);
		color.rgbBlue = static_cast<BYTE>(rgbImage[2]);

		FreeImage_SetPixelColor(bitmap, i , j, &color);
	}
	bool result = FreeImage_Save(FIF_PNG, bitmap, fileName.c_str());
	FreeImage_Unload(bitmap);
	return result;
}