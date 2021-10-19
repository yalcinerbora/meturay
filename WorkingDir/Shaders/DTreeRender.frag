#version 430
/*
	**PPG DTree Render Shader**

	File Name	: DTreeRender.vert
	Author		: Bora Yalciner
	Description	:

*/

// Definitions
#define IN_UV layout(location = 0)
#define IN_VALUE layout(location = 1)

#define OUT_COLOR layout(location = 0)
#define OUT_VALUE layout(location = 1)

#define T_IN_GRADIENT layout(binding = 0)

#define U_PERIMIETER_ON layout(location = 1)
#define U_PERIMIETER_COLOR layout(location = 2)

// Input
in IN_UV vec2 fUV;
in IN_VALUE float fValue;

// Output
out OUT_COLOR vec4 fboColor;
out OUT_VALUE float fboValue;
// Uniforms
U_PERIMIETER_ON uniform bool perimeterOn;
U_PERIMIETER_COLOR uniform vec3 perimeterColor;

// Textures
uniform T_IN_GRADIENT sampler2D tGradient;

void main(void)
{
	if(perimeterOn)
		fboColor = vec4(perimeterColor, 1.0f);
	else
	{
		fboColor = vec4(texture(tGradient, fUV).xyz, 1.0f);
		fboValue = fValue;
	}
}