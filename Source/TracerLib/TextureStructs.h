#pragma once

#include "RayLib/HitStructs.h"
#include "RayLib/Vector.h"

#include <cuda.h>
#include <cuda_fp16.h>


// METURay only supports
// up to 3D Textures and differential infor for mip fetching
// For research purposes this should be enough.
//
// Cache system statically assigns a texture array for each cached texture
// which is not stored
struct TexCoords
{    
    uint16_t    layerId;
    // Floating point values are stored as half precision formats in order to save space
    __half     u, v, w;
    __half     dx, dy, dw;
};


template <class T>
class TexturedData
{
    public:
    virtual ~TexturedData() = default;

    virtual T  Fetch(float u, float v, int w) override;
};

//// Multidimensional Texture
//template <class T>
//class Texture : public TexturedData<T>
//{
//    private:
//    cudaTextureObject_t& texture;
//
//    public:
//        // Constructos & Destructor
//    Texture();
//
//    // Interface
//    T  Fetch(float u, float v, int w) override;
//};

//class Constant


////// SFINAE the shit out of this class
////template <>
////Vector3 Texture<Vector3>::operator(float u, float v, int w)
////{}

//template <class T>
//class TextureReference
//{
//    Texture<T>& t;
//};
//
//// 
//class Radiance
//{
//    Texture<Vector3f> data;
//
//};