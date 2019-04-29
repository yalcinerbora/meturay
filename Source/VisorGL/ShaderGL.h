#pragma once
/**

Shader Class that Compiles and Binds Shaders

*/

#include <GL\glew.h>
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
		static GLuint		shaderPipelineID;

		// Properties
		GLuint				shaderID;
        ShaderType			shaderType;
		bool				valid;

		static GLenum		ShaderTypeToGL(ShaderType);
		static GLenum		ShaderTypeToGLBit(ShaderType);

	protected:

	public:
		// Constructors & Destructor
							ShaderGL();
							ShaderGL(ShaderType, const std::string& path);
							ShaderGL(ShaderGL&&);
							ShaderGL(const ShaderGL&) = delete;
		ShaderGL&			operator=(ShaderGL&&);
		ShaderGL&			operator=(const ShaderGL&) = delete;
							~ShaderGL();

		// Renderer Usage
		void				Bind();
		bool				IsValid() const;

		static void			Unbind(ShaderType);
};