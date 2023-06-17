#pragma once
/**

Shader Class that Compiles and Binds Shaders

*/

#include <glbinding/gl/gl.h>
#include <string>

enum class ShaderType
{
    VERTEX,
    TESS_C,
    TESS_E,
    GEOMETRY,
    FRAGMENT,
    COMPUTE
};

class ShaderGL
{
    private:
        // Global Variables
        static gl::GLuint   shaderPipelineID;
        static int          shaderCount;

        // Properties
        gl::GLuint          shaderID;
        ShaderType          shaderType;
        bool                valid;

        static gl::GLenum               ShaderTypeToGL(ShaderType);
        static gl::UseProgramStageMask  ShaderTypeToGLBit(ShaderType);

    protected:

    public:
        // Constructors & Destructor
                            ShaderGL();
                            ShaderGL(ShaderType, const std::u8string& path);
                            ShaderGL(ShaderGL&&) noexcept;
                            ShaderGL(const ShaderGL&) = delete;
        ShaderGL&           operator=(ShaderGL&&) noexcept;
        ShaderGL&           operator=(const ShaderGL&) = delete;
                            ~ShaderGL();

        // Renderer Usage
        void                Bind() const;
        bool                IsValid() const;

        static void         Unbind(ShaderType);
};