#version 430
/*	
	**PPG DTree Render Shader**
	
	File Name	: DTreeRender.vert 
	Author		: Bora Yalciner
	Description	:
		
*/

// Definitions
#define IN_POS layout(location = 0)
#define IN_OFFSET layout(location = 1)
#define IN_DEPTH layout(location = 2)
#define IN_RADIANCE layout(location = 3)

#define OUT_UV layout(location = 0)
#define OUT_VALUE layout(location = 1)

#define U_MAX_RADIANCE layout(location = 0)
#define U_MAX_DEPTH layout(location = 3)

// Input
// Per Vertex
in IN_POS vec2 vPos;
// Per Instance
in IN_OFFSET vec2 vOffset;
in IN_DEPTH uint vDepth;
in IN_RADIANCE float vRadiance;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_UV vec2 fUV;
out OUT_VALUE float fValue;

// Uniforms
U_MAX_RADIANCE uniform float maxRadiance;
U_MAX_DEPTH uniform uint maxDepth;

void main(void)
{
	// Calculate Scale
	float scale = pow(0.5f, vDepth);

	// Determine Gradient UV
	float u = (vRadiance / maxRadiance) / float(1 << (2 * (maxDepth - vDepth)));
	fUV = vec2(u, 0.5f);

	// Pass Value to Fragment Shader
	fValue = vRadiance;

	// [0,1] normalized position
	vec2 position = vPos * scale + vOffset;

	// Actual Position [-1, 1] (NDC)
	gl_Position = vec4(position * 2.0f - 1.0f , 0.0f, 1.0f);
}