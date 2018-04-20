/**

Camera Ray Generation Kernel

*/

#include <cstdint>
#include <cuda_runtime.h>
#include "RayLib/Vector.h"

struct CameraPerspective;
struct RayStackGMem;
struct RandomStackGMem;

__global__ void KCGenerateCameraRays(RayStackGMem gRays,
									 RandomStackGMem gRand,
									 const CameraPerspective cam,
									 const uint32_t samplePerPixel,
									 const Vector2ui resolution,
									 const Vector2ui pixelStart,
									 const Vector2ui pixelCount);
