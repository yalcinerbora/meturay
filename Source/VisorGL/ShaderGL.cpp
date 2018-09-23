#include <vector>
#include <fstream>
#include <cassert>

#include "ShaderGL.h"
#include "RayLib/Log.h"

#include <filesystem>
//namespace std { namespace filesystem = std::experimental::filesystem; }

GLuint ShaderGL::shaderPipelineID = 0;

GLenum ShaderGL::ShaderTypeToGL(ShaderType t)	
{
	static GLenum values[] = 
	{
		GL_VERTEX_SHADER,
		GL_TESS_CONTROL_SHADER,
		GL_TESS_EVALUATION_SHADER,
		GL_GEOMETRY_SHADER,
		GL_FRAGMENT_SHADER,
		GL_COMPUTE_SHADER		
	};
	return values[static_cast<int>(t)];
}

GLenum ShaderGL::ShaderTypeToGLBit(ShaderType t)
{
	static GLenum values[] =
	{
		GL_VERTEX_SHADER_BIT,
		GL_TESS_CONTROL_SHADER_BIT,
		GL_TESS_EVALUATION_SHADER_BIT,
		GL_GEOMETRY_SHADER_BIT,
		GL_FRAGMENT_SHADER_BIT,
		GL_COMPUTE_SHADER_BIT
	};
	return values[static_cast<int>(t)];
}

ShaderGL::ShaderGL()
	: shaderID(0)
{}

ShaderGL::ShaderGL(ShaderType t, const std::string& path)
	: valid(false)
	, shaderID(0)
	, shaderType(t)
{
	const std::string onlyFileName = std::filesystem::u8path(path).filename().u8string();

	std::streamoff size = std::ifstream(std::filesystem::u8path(path), std::ifstream::ate | std::ifstream::binary).tellg();
	std::vector<char> source(size + 1, 0);
	std::ifstream shaderFile(std::filesystem::u8path(path));
	assert(shaderFile.is_open());
	shaderFile.read(source.data(), source.size());

	// Create Pipeline If not Avail
	if(shaderPipelineID == 0)
	{
		glGenProgramPipelines(1, &shaderPipelineID);
		glBindProgramPipeline(shaderPipelineID);
	}

	// Compile
	const char* sourcePtr = source.data();
	shaderID = glCreateShaderProgramv(ShaderTypeToGL(shaderType), 1, (const GLchar**) &sourcePtr);

	GLint result;
	glGetProgramiv(shaderID, GL_LINK_STATUS, &result);
	// Check Errors
	if(result == GL_FALSE)
	{
		GLint blen = 0;
		glGetProgramiv(shaderID, GL_INFO_LOG_LENGTH, &blen);
		if(blen > 1)
		{
			std::vector<GLchar> log(blen);
			glGetProgramInfoLog(shaderID, blen, &blen, &log[0]);
			METU_ERROR_LOG("Shader Compilation Error on File %s :\n%s", onlyFileName.c_str(), &log[0]);
		}
	}
	else
	{
		METU_LOG("Shader Compiled Successfully. Shader ID: %d, Name: %s", shaderID, onlyFileName.c_str());
		valid = true;
	}

}

ShaderGL::ShaderGL(ShaderGL&& other)
	: shaderID(other.shaderID)
	, shaderType(other.shaderType)
	, valid(other.valid)
{
	other.shaderID = 0;
}

ShaderGL& ShaderGL::operator=(ShaderGL&& other)
{
	assert(this != &other);
	glDeleteProgram(shaderID);
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
		glDeleteProgram(shaderID);
		METU_LOG("Shader Deleted. Shader ID: %d", shaderID);
	}
	shaderID = 0;

	// We Need To Delete Pipeline Somehow Here
	// Best way is to not delete and leave it be
	// Or use some RAII but needs to be deleted before context deletion
	// else it will not get deleted anyway

	// TODO: Dont Leak OGL ShaderProgram pipeline object
}

void ShaderGL::Bind()
{
	glUseProgramStages(shaderPipelineID, ShaderTypeToGLBit(shaderType), shaderID);
	glActiveShaderProgram(shaderPipelineID, shaderID);
}

bool ShaderGL::IsValid() const
{
	return valid;
}

void ShaderGL::Unbind(ShaderType shaderType)
{
	glUseProgramStages(shaderPipelineID, ShaderTypeToGLBit(shaderType), 0);
	glActiveShaderProgram(shaderPipelineID, 0);
}