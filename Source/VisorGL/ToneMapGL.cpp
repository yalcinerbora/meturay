#include "ToneMapGL.h"
#include "RayLib/Types.h"
#include "GLConversionFunctions.h"

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

        //// Call the Kernel
        //GLuint gridX = (resolution[0] + 16 - 1) / 16;
        //GLuint gridY = (resolution[1] + 16 - 1) / 16;
        //glDispatchCompute(gridX, gridY, 1);
        //glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
    // Either gamma or not call ToneMap shader
    // since we need to transport image to SDR image
    // Post process sahder requires it to be there

    // Transfer options to GPU Memory
    glBindBuffer(GL_UNIFORM_BUFFER, tmOptionBuffer);
    glBufferData(GL_UNIFORM_BUFFER,
                 sizeof(ToneMapOptions),
                 &tmOpts, GL_STREAM_READ);

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
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

}