#pragma once

#include <memory>
#include <vector>
#include "RayLib/Vector.h"

struct SceneError;
struct SceneFileNode;

constexpr const char* NodeSphereName = "nodeSphere";
constexpr const char* NodeTriangleName = "nodeTriangle";

class SurfaceDataLoaderI
{
	public:
		virtual							~SurfaceDataLoaderI() = default;		

		// Size Determination
		virtual size_t					PrimitiveCount() const = 0;
		virtual size_t					PrimitiveDataSize(const std::string& primitiveDataType) const = 0;

		// Load Functionality
		virtual const char*				SufaceDataFileExt() const = 0;
		virtual const uint32_t			SurfaceDataId() const = 0;
		
		//
		virtual SceneError				LoadPrimitiveData(float*,
														  const std::string& primitiveDataType) = 0;
		virtual SceneError				LoadPrimitiveData(int*,
														  const std::string& primitiveDataType) = 0;
		virtual SceneError				LoadPrimitiveData(unsigned int*,
														  const std::string& primitiveDataType) = 0;
};

namespace SurfaceDataIO
{
	std::unique_ptr<SurfaceDataLoaderI>		GenSurfaceDataLoader(const SceneFileNode& properties,
																 double time);	
}
