#include "TextureReference.cuh"

template class ConstantRef<1, float>;
template class ConstantRef<1, Vector2>;
template class ConstantRef<1, Vector3>;
template class ConstantRef<1, Vector4>;

template class ConstantRef<2, float>;
template class ConstantRef<2, Vector2>;
template class ConstantRef<2, Vector3>;
template class ConstantRef<2, Vector4>;

template class ConstantRef<3, float>;
template class ConstantRef<3, Vector2>;
template class ConstantRef<3, Vector3>;
template class ConstantRef<3, Vector4>;

template class TextureRef<1, float>;
template class TextureRef<1, Vector2>;
template class TextureRef<1, Vector3>;
template class TextureRef<1, Vector4>;

//template class TextureRef<2, float>;
template class TextureRef<2, Vector2>;
template class TextureRef<2, Vector3>;
template class TextureRef<2, Vector4>;

template class TextureRef<3, float>;
template class TextureRef<3, Vector2>;
template class TextureRef<3, Vector3>;
template class TextureRef<3, Vector4>;