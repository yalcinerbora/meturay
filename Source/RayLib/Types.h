#pragma once
/**

*/

#include <cstdint>

typedef unsigned char Byte;

static constexpr uint16_t NullBatchId = 0;

static constexpr uint16_t BYTE_BITS = 8;

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

    R8_SNORM,
    RG8_SNORM,
    RGB8_SNORM,
    RGBA8_SNORM,

    R16_SNORM,
    RG16_SNORM,
    RGB16_SNORM,
    RGBA16_SNORM,

    R_HALF,
    RG_HALF,
    RGB_HALF,
    RGBA_HALF,

    R_FLOAT,
    RG_FLOAT,
    RGB_FLOAT,
    RGBA_FLOAT,

    // Graphics Related Compressed Images
    // https://docs.microsoft.com/en-us/windows/win32/direct3d11/texture-block-compression-in-direct3d-11
    // CUDA also support these but it is pain in the ass to properly load into
    // textureObjects
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g6b3a50368a0aa592f65e928adca9b929
    BC1_U,
    BC2_U,
    BC3_U,
    BC4_U,
    BC4_S,
    BC5_U,
    BC5_S,
    BC6H_U,
    BC6H_S,
    BC7_U,

    END
};

//enum class DataType
//{
//    // Float Data Types
//    HALF_1,             // one component IEEE 754 (16-bit) floating point number
//    HALF_2,             // two component                   ""
//    HALF_3,             // three component                 ""
//    HALF_4,             // four component                  ""
//
//    FLOAT_1,            // one component IEEE 754 (32-bit) floating point number
//    FLOAT_2,            // two component                   ""
//    FLOAT_3,            // three component                 ""
//    FLOAT_4,            // four component                  ""
//
//    DOUBLE_1,           // one component IEEE 754 (64-bit) floating point number
//    DOUBLE_2,           // two component                   ""
//    DOUBLE_3,           // three component                 ""
//    DOUBLE_4,           // four component                  ""
//
//    // Nobody will ever use this in this decade but w/e
//    QUADRUPLE_1,        // one component IEEE 754 (128-bit) floating point number
//    QUADRUPLE_2,        // two component                   ""
//    QUADRUPLE_3,        // three component                 ""
//    QUADRUPLE_4,        // four component                  ""
//
//    // Integers
//    // 8-bit
//    INT8_1,
//    INT8_2,
//    INT8_3,
//    INT8_4,
//
//    UINT8_1,
//    UINT8_2,
//    UINT8_3,
//    UINT8_4,
//
//    // 16-bit
//    INT16_1,
//    INT16_2,
//    INT16_3,
//    INT16_4,
//
//    UINT16_1,
//    UINT16_2,
//    UINT16_3,
//    UINT16_4,
//
//    // 32-bit
//    INT32_1,
//    INT32_2,
//    INT32_3,
//    INT32_4,
//
//    UINT32_1,
//    UINT32_2,
//    UINT32_3,
//    UINT32_4,
//
//    // 64-bit
//    INT64_1,
//    INT64_2,
//    INT64_3,
//    INT64_4,
//
//    UINT64_1,
//    UINT64_2,
//    UINT64_3,
//    UINT64_4,
//
//    // Normalized Data Types (DX UNORM or NORM)
//    // Fixed Point
//    // Definition of UNORM/NORM can be found here
//    // http://msdn.microsoft.com/en-us/library/windows/desktop/dd607323(v=vs.85).aspx
//
//    // 8-bit
//    NORM8_1,
//    NORM8_2,
//    NORM8_3,
//    NORM8_4,
//
//    UNORM8_1,
//    UNORM8_2,
//    UNORM8_3,
//    UNORM8_4,
//
//    // 16-bit
//    NORM16_1,
//    NORM16_2,
//    NORM16_3,
//    NORM16_4,
//
//    UNORM16_1,
//    UNORM16_2,
//    UNORM16_3,
//    UNORM16_4,
//
//    // 32-bit
//    NORM32_1,
//    NORM32_2,
//    NORM32_3,
//    NORM32_4,
//
//    UNORM32_1,
//    UNORM32_2,
//    UNORM32_3,
//    UNORM32_4,
//
//    END
//};