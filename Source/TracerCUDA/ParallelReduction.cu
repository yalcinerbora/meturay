#include "ParallelReduction.cuh"

//// Cluster Definitions ARRAY
//#define DEFINE_REDUCE_ARRAY_BOTH(type, func) \
//    DEFINE_REDUCE_ARRAY_SINGLE(type, func, cudaMemcpyDeviceToHost) \
//    DEFINE_REDUCE_ARRAY_SINGLE(type, func, cudaMemcpyDeviceToDevice)
//
//#define DEFINE_REDUCE_ARRAY_ALL(type) \
//    DEFINE_REDUCE_ARRAY_BOTH(type, ReduceAdd) \
//    DEFINE_REDUCE_ARRAY_BOTH(type, ReduceSubtract) \
//    DEFINE_REDUCE_ARRAY_BOTH(type, ReduceMultiply) \
//    DEFINE_REDUCE_ARRAY_BOTH(type, ReduceDivide) \
//    DEFINE_REDUCE_ARRAY_BOTH(type, ReduceMin) \
//    DEFINE_REDUCE_ARRAY_BOTH(type, ReduceMax)
//
//// Cluster Definitions TEXTURE
//#define DEFINE_REDUCE_TEXTURE_BOTH(type, func) \
//    DEFINE_REDUCE_TEXTURE_SINGLE(type, func, cudaMemcpyDeviceToHost) \
//    DEFINE_REDUCE_TEXTURE_SINGLE(type, func, cudaMemcpyDeviceToDevice)
//
//#define DEFINE_REDUCE_TEXTURE_ALL(type) \
//    DEFINE_REDUCE_TEXTURE_BOTH(type, ReduceAdd) \
//    DEFINE_REDUCE_TEXTURE_BOTH(type, ReduceSubtract) \
//    DEFINE_REDUCE_TEXTURE_BOTH(type, ReduceMultiply) \
//    DEFINE_REDUCE_TEXTURE_BOTH(type, ReduceDivide) \
//    DEFINE_REDUCE_TEXTURE_BOTH(type, ReduceMin) \
//    DEFINE_REDUCE_TEXTURE_BOTH(type, ReduceMax)
//
//// Integral Types
//DEFINE_REDUCE_ARRAY_ALL(int)
//DEFINE_REDUCE_ARRAY_ALL(unsigned int)
//DEFINE_REDUCE_ARRAY_ALL(float)
//DEFINE_REDUCE_ARRAY_ALL(double)
//DEFINE_REDUCE_ARRAY_ALL(int64_t)
//DEFINE_REDUCE_ARRAY_ALL(uint64_t)
//
//// Vector Types
//DEFINE_REDUCE_ARRAY_ALL(Vector2f)
//DEFINE_REDUCE_ARRAY_ALL(Vector2d)
//DEFINE_REDUCE_ARRAY_ALL(Vector2i)
//DEFINE_REDUCE_ARRAY_ALL(Vector2ui)
//
//DEFINE_REDUCE_ARRAY_ALL(Vector3f)
//DEFINE_REDUCE_ARRAY_ALL(Vector3d)
//DEFINE_REDUCE_ARRAY_ALL(Vector3i)
//DEFINE_REDUCE_ARRAY_ALL(Vector3ui)
//
//DEFINE_REDUCE_ARRAY_ALL(Vector4f)
//DEFINE_REDUCE_ARRAY_ALL(Vector4d)
//DEFINE_REDUCE_ARRAY_ALL(Vector4i)
//DEFINE_REDUCE_ARRAY_ALL(Vector4ui)
//
//// Matrix Types
//DEFINE_REDUCE_ARRAY_ALL(Matrix2x2f)
//DEFINE_REDUCE_ARRAY_ALL(Matrix2x2d)
//DEFINE_REDUCE_ARRAY_ALL(Matrix2x2i)
//DEFINE_REDUCE_ARRAY_ALL(Matrix2x2ui)
//
//DEFINE_REDUCE_ARRAY_ALL(Matrix3x3f)
//DEFINE_REDUCE_ARRAY_ALL(Matrix3x3d)
//DEFINE_REDUCE_ARRAY_ALL(Matrix3x3i)
//DEFINE_REDUCE_ARRAY_ALL(Matrix3x3ui)
//
//DEFINE_REDUCE_ARRAY_ALL(Matrix4x4f)
//DEFINE_REDUCE_ARRAY_ALL(Matrix4x4d)
//DEFINE_REDUCE_ARRAY_ALL(Matrix4x4i)
//DEFINE_REDUCE_ARRAY_ALL(Matrix4x4ui)
//
//// Quaternion Types
//DEFINE_REDUCE_ARRAY_BOTH(QuatF, ReduceMultiply)
//DEFINE_REDUCE_ARRAY_BOTH(QuatD, ReduceMultiply)
//
//// Texture Types
//DEFINE_REDUCE_TEXTURE_ALL(float)
//DEFINE_REDUCE_TEXTURE_ALL(float2)
//DEFINE_REDUCE_TEXTURE_ALL(float4)