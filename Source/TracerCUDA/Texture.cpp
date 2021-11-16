#include "Texture.cuh"

TextureCubeI::TextureCubeI(TextureCubeI&& other)
    : DeviceLocalMemoryI(other.currentDevice)
    , texture(other.texture)
    , dimensions(other.dimensions)
    , channelCount(other.channelCount)
{
    other.texture = 0;
    other.dimensions = Zero2ui;
    other.channelCount = 0;
}

TextureCubeI& TextureCubeI::operator=(TextureCubeI&& other)
{
    assert(this != &other);
    texture = other.texture;
    dimensions = other.dimensions;
    other.texture = 0;
    other.dimensions = Zero2ui;
    return *this;
}

template class Texture<1, float>;
template class Texture<1, Vector2>;
template class Texture<1, Vector4>;
template class Texture<1, int>;
template class Texture<1, int2>;
template class Texture<1, int4>;
template class Texture<1, short>;
template class Texture<1, short2>;
template class Texture<1, short4>;
template class Texture<1, char>;
template class Texture<1, char2>;
template class Texture<1, char4>;
template class Texture<1, unsigned int>;
template class Texture<1, uint2>;
template class Texture<1, uint4>;
template class Texture<1, unsigned short>;
template class Texture<1, ushort2>;
template class Texture<1, ushort4>;
template class Texture<1, unsigned char>;
template class Texture<1, uchar2>;
template class Texture<1, uchar4>;

template class Texture<2, float>;
template class Texture<2, Vector2>;
template class Texture<2, Vector4>;
template class Texture<2, int>;
template class Texture<2, int2>;
template class Texture<2, int4>;
template class Texture<2, short>;
template class Texture<2, short2>;
template class Texture<2, short4>;
template class Texture<2, char>;
template class Texture<2, char2>;
template class Texture<2, char4>;
template class Texture<2, unsigned int>;
template class Texture<2, uint2>;
template class Texture<2, uint4>;
template class Texture<2, unsigned short>;
template class Texture<2, ushort2>;
template class Texture<2, ushort4>;
template class Texture<2, unsigned char>;
template class Texture<2, uchar2>;
template class Texture<2, uchar4>;

template class Texture<3, float>;
template class Texture<3, Vector2>;
template class Texture<3, Vector4>;
template class Texture<3, int>;
template class Texture<3, int2>;
template class Texture<3, int4>;
template class Texture<3, short>;
template class Texture<3, short2>;
template class Texture<3, short4>;
template class Texture<3, char>;
template class Texture<3, char2>;
template class Texture<3, char4>;
template class Texture<3, unsigned int>;
template class Texture<3, uint2>;
template class Texture<3, uint4>;
template class Texture<3, unsigned short>;
template class Texture<3, ushort2>;
template class Texture<3, ushort4>;
template class Texture<3, unsigned char>;
template class Texture<3, uchar2>;
template class Texture<3, uchar4>;

template class TextureArray<1, float>;
template class TextureArray<1, Vector2>;
template class TextureArray<1, Vector4>;
template class TextureArray<1, int>;
template class TextureArray<1, int2>;
template class TextureArray<1, int4>;
template class TextureArray<1, short>;
template class TextureArray<1, short2>;
template class TextureArray<1, short4>;
template class TextureArray<1, char>;
template class TextureArray<1, char2>;
template class TextureArray<1, char4>;
template class TextureArray<1, unsigned int>;
template class TextureArray<1, uint2>;
template class TextureArray<1, uint4>;
template class TextureArray<1, unsigned short>;
template class TextureArray<1, ushort2>;
template class TextureArray<1, ushort4>;
template class TextureArray<1, unsigned char>;
template class TextureArray<1, uchar2>;
template class TextureArray<1, uchar4>;

template class TextureArray<2, float>;
template class TextureArray<2, Vector2>;
template class TextureArray<2, Vector4>;
template class TextureArray<2, int>;
template class TextureArray<2, int2>;
template class TextureArray<2, int4>;
template class TextureArray<2, short>;
template class TextureArray<2, short2>;
template class TextureArray<2, short4>;
template class TextureArray<2, char>;
template class TextureArray<2, char2>;
template class TextureArray<2, char4>;
template class TextureArray<2, unsigned int>;
template class TextureArray<2, uint2>;
template class TextureArray<2, uint4>;
template class TextureArray<2, unsigned short>;
template class TextureArray<2, ushort2>;
template class TextureArray<2, ushort4>;
template class TextureArray<2, unsigned char>;
template class TextureArray<2, uchar2>;
template class TextureArray<2, uchar4>;

template class TextureCube<float>;
template class TextureCube<Vector2>;
template class TextureCube<Vector4>;
template class TextureCube<int>;
template class TextureCube<int2>;
template class TextureCube<int4>;
template class TextureCube<short>;
template class TextureCube<short2>;
template class TextureCube<short4>;
template class TextureCube<char>;
template class TextureCube<char2>;
template class TextureCube<char4>;
template class TextureCube<unsigned int>;
template class TextureCube<uint2>;
template class TextureCube<uint4>;
template class TextureCube<unsigned short>;
template class TextureCube<ushort2>;
template class TextureCube<ushort4>;
template class TextureCube<unsigned char>;
template class TextureCube<uchar2>;
template class TextureCube<uchar4>;