#pragma once

#include <gl/glew.h>

#include "RayLib/Vector.h"
#include "Structs.h"
#include "ShaderGL.h"

enum class PixelFormat;

class ToneMapGL
{
    private:
        ShaderGL                    compLumReduce;
        ShaderGL                    compToneMap;

        GLuint                      tmOptionBuffer;
        GLuint                      luminanceBuffer;

        // Uniforms
        static constexpr GLenum     U_RES = 0;

        // Uniform Buffers
        static constexpr GLenum     UB_LUM_DATA = 0;
        static constexpr GLenum     UB_TM_PARAMS = 1;
        // Shader Storage Buffers
        static constexpr GLenum     SSB_OUTPUT = 0;
        // Texures
        static constexpr GLenum     T_IN_HDR_IMAGE = 0;
        // Images
        static constexpr GLenum     I_OUT_SDR_IMAGE = 0;

        // GL Image of the luminance buffer
        struct LumBufferGL
        {
            float                   outMaxLum;
            float                   outAvgLum;
        };

    protected:
    public:
        // Constructors & Destructor
                                ToneMapGL();
                                ToneMapGL(const ToneMapGL&) = delete;
                                ToneMapGL(ToneMapGL&&) = delete;
        ToneMapGL&              operator=(ToneMapGL) = delete;
        ToneMapGL&              operator=(ToneMapGL&&);
                                ~ToneMapGL();

        void                    ToneMap(GLuint sdrTexture, 
                                        const PixelFormat sdrPixelFormat,
                                        const GLuint hdrTexture,
                                        const ToneMapOptions& options,
                                        const Vector2i& resolution);
};

inline ToneMapGL::ToneMapGL()
{
    //compLumReduce = ShaderGL(ShaderType::COMPUTE, u8"Shaders/LumReduction.comp");
    //compToneMap = ShaderGL(ShaderType::COMPUTE, u8"Shaders/TonemapGamma.comp");
}

inline ToneMapGL::~ToneMapGL()
{
    compLumReduce = ShaderGL();
    compToneMap = ShaderGL();
}

inline ToneMapGL& ToneMapGL::operator=(ToneMapGL&&)
{
    return *this;
}