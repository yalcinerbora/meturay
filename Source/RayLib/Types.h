#pragma once
/**

*/
typedef unsigned char byte;

enum class PixelFormat
{
	R8_UNORM,
	RG8_UNORM,
	RGB8_UNORM,
	RGBA8_UNORM,

	R16_UNORM,
	RG16_UNORM,
	RGB16_UNORM,
	RGBA16_UNORM,
	
	R_HALF,
	RG_HALF,
	RGB_HALF,
	RGBA_HALF,

	R_FLOAT,
	RG_FLOAT,
	RGB_FLOAT,
	RGBA_FLOAT,

	END
};
