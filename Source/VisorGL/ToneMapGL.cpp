#include "ToneMapGL.h"
#include "GLConversionFunctions.h"
#include "RayLib/Log.h"
#include <vector>

void ToneMapGL::ToneMap(GLuint sdrTexture,
                        const PixelFormat sdrPixelFormat,
                        const GLuint hdrTexture,
                        const ToneMapOptions& tmOpts,
                        const Vector2i& resolution)
{
    // Check options if tone map is requested update
    // max/avg luminance
    if(tmOpts.doToneMap)
    {
        // Clear Luminance Buffer
        glBindBuffer(GL_COPY_WRITE_BUFFER, luminanceBuffer);
        glClearBufferData(GL_COPY_WRITE_BUFFER, GL_R8, GL_RED,
                          GL_BYTE, nullptr);

        // Unbind Luminance buffer as UBO just to be sure
        glBindBufferBase(GL_UNIFORM_BUFFER, UB_LUM_DATA, 0);
        // Bind the Shader
        compLumReduce.Bind();
        // Bind Uniforms
        glUniform2iv(U_RES, 1, static_cast<const int*>(resolution));
        // Bind SSBO
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSB_OUTPUT, luminanceBuffer);
        // Bind Textures
        // Bind HDR Texture
        glActiveTexture(GL_TEXTURE0 + T_IN_HDR_IMAGE);
        glBindTexture(GL_TEXTURE_2D, hdrTexture);

        // Call the Kernel
        GLuint workCount = resolution[1] * resolution[0];
        GLuint gridX = (workCount + 256 - 1) / 256;
        glDispatchCompute(gridX, 1, 1);
        glMemoryBarrier(GL_UNIFORM_BARRIER_BIT |
                        GL_SHADER_STORAGE_BARRIER_BIT);

        // Now Call simple Average Kernel
        compAvgDivisor.Bind();
        // Bind Uniforms
        glUniform2iv(U_RES, 1, static_cast<const int*>(resolution));
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_UNIFORM_BARRIER_BIT);

        //// Debug check of the reduced values
        //Vector2f v;
        //glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        //glBindBuffer(GL_COPY_READ_BUFFER, luminanceBuffer);
        //glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(Vector2),
        //                   &v);
        //METU_LOG("max {:f}, avg {:f}", v[0], v[1]);
    }
    // Either gamma or not, call ToneMap shader
    // since we need to transport image to SDR image
    // Post process shader requires it to be there

    // Align Padding for UBO
    TMOBufferGL tmOptsGL;
    tmOptsGL.doToneMap = static_cast<uint32_t>(tmOpts.doToneMap);
    tmOptsGL.doGamma = static_cast<uint32_t>(tmOpts.doGamma);
    tmOptsGL.doKeyAdjust = static_cast<uint32_t>(tmOpts.doKeyAdjust);
    tmOptsGL.gammaValue = tmOpts.gamma;
    tmOptsGL.burnRatio = tmOpts.burnRatio;
    tmOptsGL.key = tmOpts.key;

    // Transfer options to GPU Memory
    glBindBuffer(GL_UNIFORM_BUFFER, tmOptionBuffer);
    glBufferSubData(GL_UNIFORM_BUFFER, 0,
                    sizeof(TMOBufferGL),
                    &tmOptsGL);

    // Bind Shader
    compToneMap.Bind();
    // Bind Uniforms
    glUniform2iv(U_RES, 1, static_cast<const int*>(resolution));
    // Bind UBOs
    // Bind UBO for the max/avg Luminance
    glBindBufferBase(GL_UNIFORM_BUFFER, UB_LUM_DATA, luminanceBuffer);
    // Bind UBO for Tone Map Parameters
    glBindBufferBase(GL_UNIFORM_BUFFER, UB_TM_PARAMS, tmOptionBuffer);
    // Bind Textures
    // Bind HDR Texture
    glActiveTexture(GL_TEXTURE0 + T_IN_HDR_IMAGE);
    glBindTexture(GL_TEXTURE_2D, hdrTexture);
    // Bind Images
    // Bind SDR Texture as Image
    glBindImageTexture(I_OUT_SDR_IMAGE, sdrTexture,
                       0, false, 0, GL_WRITE_ONLY,
                       PixelFormatToSizedGL(sdrPixelFormat));

    // Call the Kernel
    GLuint gridX = (resolution[0] + 16 - 1) / 16;
    GLuint gridY = (resolution[1] + 16 - 1) / 16;
    glDispatchCompute(gridX, gridY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                    GL_TEXTURE_FETCH_BARRIER_BIT);
}