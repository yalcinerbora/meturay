#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "RayLib/Texture.h"



TEST(TextureCPU, Construction)
{
	TextureData d;
	Sampler s;

	Texture2 test(d, s);

	Vector3 v3(1.0f, 1.0f, 1.0f);
	Vector2 v2(1.0f, 1.0f);

	Vector3i v3i(1, 1, 1);
	Vector2i v2i(1, 1);

}