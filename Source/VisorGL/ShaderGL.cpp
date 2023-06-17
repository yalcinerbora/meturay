#include <vector>
#include <fstream>
#include <cassert>

#include "ShaderGL.h"

#include "RayLib/Log.h"
#include "RayLib/UTF8StringConversion.h"
#include "RayLib/FileSystemUtility.h"

#include <filesystem>

gl::GLuint ShaderGL::shaderPipelineID = 0;
int ShaderGL::shaderCount = 0;

gl::GLenum ShaderGL::ShaderTypeToGL(ShaderType t)
{
    static gl::GLenum values[] =
    {
        gl::GL_VERTEX_SHADER,
        gl::GL_TESS_CONTROL_SHADER,
        gl::GL_TESS_EVALUATION_SHADER,
        gl::GL_GEOMETRY_SHADER,
        gl::GL_FRAGMENT_SHADER,
        gl::GL_COMPUTE_SHADER
    };
    return values[static_cast<int>(t)];
}

gl::UseProgramStageMask ShaderGL::ShaderTypeToGLBit(ShaderType t)
{
    static gl::UseProgramStageMask values[] =
    {
        gl::GL_VERTEX_SHADER_BIT,
        gl::GL_TESS_CONTROL_SHADER_BIT,
        gl::GL_TESS_EVALUATION_SHADER_BIT,
        gl::GL_GEOMETRY_SHADER_BIT,
        gl::GL_FRAGMENT_SHADER_BIT,
        gl::GL_COMPUTE_SHADER_BIT
    };
    return values[static_cast<int>(t)];
}

ShaderGL::ShaderGL()
    : shaderID(0)
    , shaderType(ShaderType::COMPUTE)
    , valid(false)
{}

ShaderGL::ShaderGL(ShaderType t, const std::u8string& path)
    : shaderID(0)
    , shaderType(t)
    , valid(false)
{
    const std::u8string onlyFileName = std::filesystem::path(path).filename().u8string();
    std::filesystem::path execPath = Utility::CurrentExecPath();
    std::filesystem::path filePath(path);
    std::filesystem::path fullPath = execPath / filePath;


    std::streamoff size = std::ifstream(fullPath,
                                        std::ifstream::ate |
                                        std::ifstream::binary).tellg();
    std::vector<char> source(size + 1, 0);
    std::ifstream shaderFile = std::ifstream(fullPath);
    assert(shaderFile.is_open());
    shaderFile.read(source.data(), source.size());

    // Create Pipeline If not Avail
    if(shaderPipelineID == 0)
    {
        gl::glGenProgramPipelines(1, &shaderPipelineID);
        gl::glBindProgramPipeline(shaderPipelineID);
    }

    // Compile
    const char* sourcePtr = source.data();
    shaderID = glCreateShaderProgramv(ShaderTypeToGL(shaderType), 1, (const gl::GLchar**) &sourcePtr);

    gl::GLint result;
    gl::glGetProgramiv(shaderID, gl::GL_LINK_STATUS, &result);
    // Check Errors
    if(result == gl::GL_FALSE)
    {
        gl::GLint blen = 0;
        gl::glGetProgramiv(shaderID, gl::GL_INFO_LOG_LENGTH, &blen);
        if(blen > 1)
        {
            static_assert(std::is_same_v<char, gl::GLchar>, "TypeMismatch: GLchar != char");
            std::string log(blen + 1, '\0');
            gl::glGetProgramInfoLog(shaderID, blen, &blen, log.data());
            METU_ERROR_LOG("Shader Compilation Error on File {:s} :\n{:s}", Utility::CopyU8ToString(onlyFileName), log);
        }
    }
    else
    {
        METU_LOG("Shader Compiled Successfully. Shader ID: {:d}, Name: {:s}", shaderID, Utility::CopyU8ToString(onlyFileName));
        valid = true;
    }
    shaderCount++;
}

ShaderGL::ShaderGL(ShaderGL&& other) noexcept
    : shaderID(other.shaderID)
    , shaderType(other.shaderType)
    , valid(other.valid)
{
    other.shaderID = 0;
}

ShaderGL& ShaderGL::operator=(ShaderGL&& other) noexcept
{
    assert(this != &other);
    gl::glDeleteProgram(shaderID);
    if(shaderID != 0)
    {
        METU_LOG("Shader Deleted. Shader ID: {:d}", shaderID);
        // Deleting shader pipeline if no shader is left
        shaderCount--;
        if(shaderCount == 0)
        {
            gl::glBindProgramPipeline(0);
            gl::glDeleteProgramPipelines(1, &shaderPipelineID);
            shaderPipelineID = 0;
        }
    }
    shaderID = other.shaderID;
    shaderType = other.shaderType;
    valid = other.valid;
    other.shaderID = 0;
    return *this;
}

ShaderGL::~ShaderGL()
{
    if(shaderID)
    {
        gl::glDeleteProgram(shaderID);
        METU_LOG("Shader Deleted. Shader ID: {:d}", shaderID);

        // Deleting shader pipeline if no shader is left
        shaderCount--;
        if(shaderCount == 0)
        {
            gl::glBindProgramPipeline(0);
            gl::glDeleteProgramPipelines(1, &shaderPipelineID);
            shaderPipelineID = 0;
        }
    }
    shaderID = 0;
}

void ShaderGL::Bind() const
{
    gl::glUseProgramStages(shaderPipelineID, ShaderTypeToGLBit(shaderType), shaderID);
    gl::glActiveShaderProgram(shaderPipelineID, shaderID);
}

bool ShaderGL::IsValid() const
{
    return valid;
}

void ShaderGL::Unbind(ShaderType shaderType)
{
    gl::glUseProgramStages(shaderPipelineID, ShaderTypeToGLBit(shaderType), 0);
    gl::glActiveShaderProgram(shaderPipelineID, 0);
}