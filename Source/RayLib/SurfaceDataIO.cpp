#include "SurfaceDataIO.h"


std::unique_ptr<SurfaceDataLoaderI> SurfaceDataIO::GenSurfaceDataLoader(const SceneFileNode& properties)
{
	return std::unique_ptr<SurfaceDataLoaderI>(nullptr);
}