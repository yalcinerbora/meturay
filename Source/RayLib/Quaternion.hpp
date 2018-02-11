#pragma once

__device__ __host__
constexpr Quaternion::Quaternion()
	: vec(1.0f, 0.0f, 0.0f, 0.0f)
{}

__device__ __host__
constexpr Quaternion::Quaternion(float w, float x, float y, float z)
: vec(w, x, y, z)
{}

__device__ __host__
constexpr Quaternion::Quaternion(const float* v)
	: vec(v)
{}

__device__ __host__ Quaternion::Quaternion(float angle, const Vector3& axis)
{
	// TODO:
}
