
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

template class TexRef<1, float>;
template class TexRef<1, Vector2>;
template class TexRef<1, Vector3>;
template class TexRef<1, Vector4>;

template class TexRef<2, float>;
template class TexRef<2, Vector2>;
template class TexRef<2, Vector3>;
template class TexRef<2, Vector4>;

template class TexRef<3, float>;
template class TexRef<3, Vector2>;
template class TexRef<3, Vector3>;
template class TexRef<3, Vector4>;