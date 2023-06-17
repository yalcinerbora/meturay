#pragma once

#include <glbinding/gl/gl.h>

#include "RayLib/Types.h"
#include "RayLib/Vector.h"
#include "Structs.h"
#include "ShaderGL.h"

enum class PixelFormat;

class ToneMapGL
{
    private:
        ShaderGL                        compLumReduce;
        ShaderGL                        compToneMap;
        ShaderGL                        compAvgDivisor;

        gl::GLuint                      tmOptionBuffer;
        gl::GLuint                      luminanceBuffer;

        // Uniforms
        static constexpr gl::GLuint     U_RES = 0;

        // Uniform Buffers
        static constexpr gl::GLuint     UB_LUM_DATA = 0;
        static constexpr gl::GLuint     UB_TM_PARAMS = 1;
        // Shader Storage Buffers
        static constexpr gl::GLuint     SSB_OUTPUT = 0;
        // Textures
        static constexpr gl::GLuint     T_IN_HDR_IMAGE = 0;
        // Images
        static constexpr gl::GLuint     I_OUT_SDR_IMAGE = 0;

        // GL Image of the luminance buffer
        #pragma pack(push, 1)
        struct LumBufferGL
        {
            float       outMaxLum;
            float       outAvgLum;
        };
        struct TMOBufferGL
        {
            // OGL stores bools as uint
            // OGL Spec 4.5 page 137
            uint32_t    doToneMap;
            uint32_t    doGamma;
            uint32_t    doKeyAdjust;
            float       gammaValue;
            float       burnRatio;
            float       key;
        };
        #pragma pack(pop)

    protected:
    public:
        // Constructors & Destructor
                                ToneMapGL(bool isOGLContextActive = false);
                                ToneMapGL(const ToneMapGL&) = delete;
                                ToneMapGL(ToneMapGL&&) = delete;
        ToneMapGL&              operator=(const ToneMapGL&) = delete;
        ToneMapGL&              operator=(ToneMapGL&&);
                                ~ToneMapGL();

        void                    ToneMap(gl::GLuint sdrTexture,
                                        const PixelFormat sdrPixelFormat,
                                        const gl::GLuint hdrTexture,
                                        const ToneMapOptions& options,
                                        const Vector2i& resolution);
};

inline ToneMapGL::ToneMapGL(bool isOGLContextActive)
    : tmOptionBuffer(0)
    , luminanceBuffer(0)
{
    if(!isOGLContextActive) return;

    // Compile Shaders
    compLumReduce = ShaderGL(ShaderType::COMPUTE, u8"Shaders/LumReduction.comp");
    compToneMap = ShaderGL(ShaderType::COMPUTE, u8"Shaders/TonemapAndGamma.comp");
    compAvgDivisor = ShaderGL(ShaderType::COMPUTE, u8"Shaders/AvgDivisor.comp");

    //size_t lumBufferSizeDebug = sizeof(LumBufferGL) + sizeof(float) * 64;
    // Allocate Buffers
    gl::glGenBuffers(1, &luminanceBuffer);
    gl::glBindBuffer(gl::GL_COPY_WRITE_BUFFER, luminanceBuffer);

    //glBufferData(GL_COPY_WRITE_BUFFER,
    //             sizeof(LumBufferGL), nullptr,
    //             GL_DYNAMIC_DRAW);

    gl::glBufferStorage(gl::GL_COPY_WRITE_BUFFER,
                        sizeof(LumBufferGL), nullptr,
                        // This buffer is GPU only so no flags required
                        // Intel UHD Graphics does not call
                        // "glClearBufferData" over non-dynamic
                        // storages so I've set the bit
                        gl::GL_DYNAMIC_STORAGE_BIT);

    gl::glGenBuffers(1, &tmOptionBuffer);
    gl::glBindBuffer(gl::GL_COPY_WRITE_BUFFER, tmOptionBuffer);
    gl::glBufferStorage(gl::GL_COPY_WRITE_BUFFER,
                        sizeof(TMOBufferGL), nullptr,
                        gl::GL_MAP_WRITE_BIT |
                        gl::GL_DYNAMIC_STORAGE_BIT);
}

inline ToneMapGL::~ToneMapGL()
{
    compLumReduce = ShaderGL();
    compToneMap = ShaderGL();
    compAvgDivisor = ShaderGL();

    gl::glDeleteBuffers(1, &luminanceBuffer);
    gl::glDeleteBuffers(1, &tmOptionBuffer);
}

inline ToneMapGL& ToneMapGL::operator=(ToneMapGL&& other)
{
    assert(this != &other);
    compLumReduce = std::move(other.compLumReduce);
    compToneMap = std::move(other.compToneMap);
    compAvgDivisor = std::move(other.compAvgDivisor);

    luminanceBuffer = other.luminanceBuffer;
    tmOptionBuffer = other.tmOptionBuffer;
    other.luminanceBuffer = 0;
    other.tmOptionBuffer = 0;
    return *this;
}