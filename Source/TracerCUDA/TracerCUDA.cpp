#include "TracerCUDA.h"

void TracerCUDA::THRDAssignScene(const SceneI&)
{

}

void TracerCUDA::THRDSetParams(const TracerParameters&)
{

}

void TracerCUDA::THRDGenerateSceneAccelerator()
{

}

void TracerCUDA::THRDGenerateAccelerator(uint32_t objId)
{

}

void TracerCUDA::THRDAssignImageSegment(const Vector2ui& pixelStart,
										const Vector2ui& pixelEnd)
{

}

void TracerCUDA::THRDAssignAllMaterials()
{

}

void TracerCUDA::THRDAssignMaterial(uint32_t matId)
{

}

void TracerCUDA::THRDLoadMaterial(uint32_t matId)
{

}

void TracerCUDA::THRDUnloadMaterial(uint32_t matId)
{

}

void TracerCUDA::THRDGenerateCameraRays(const CameraPerspective& camera,
										const uint32_t samplePerPixel)
{

}

void TracerCUDA::THRDHitRays()
{

}

void TracerCUDA::THRDGetMaterialRays(const RayRecodCPU&, uint32_t rayCount, uint32_t matId)
{

}

void TracerCUDA::THRDAddMaterialRays(const ConstRayRecodCPU&, uint32_t rayCount, uint32_t matId)
{

}

void TracerCUDA::THRDBounceRays()
{

}

uint32_t TracerCUDA::THRDRayCount()
{
	return 0;
}

TracerCUDA::TracerCUDA()
{

}

TracerCUDA::~TracerCUDA()
{

}

void TracerCUDA::Initialize()
{

}