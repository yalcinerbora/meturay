#version 430
/*
	**PPG DTree Render Shader**
	
	File Name	: DTreeRender.vert 
	Author		: Bora Yalciner
	Description	:

*/

// Definitions
#define IN_UV layout(location = 0)

#define OUT_COLOR layout(location = 0)

#define T_IN_GRADIENT layout(binding = 0)

#define U_PERIMIETER_ON layout(location = 1)
#define U_PERIMIETER_COLOR layout(location = 2)

// Input
in IN_UV vec2 fUV;

// Output
out OUT_COLOR vec4 fboColor;
// Uniforms
uniform U_PERIMIETER_ON bool perimeterOn;
uniform U_PERIMIETER_COLOR vec3 perimeterColor;

// Textures
uniform T_IN_GRADIENT sampler2D tGradient;

void main(void)
{
	if(perimeterOn)
		fboColor = vec4(texture(tGradient, fUV).xyz, 1.0f);
	else
		fboColor = vec4(perimeterColor, 1.0f);
}