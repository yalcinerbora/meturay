#include "SurfaceDataIO.h"
#include "SceneIO.h"
//#include "SceneFileNode.h"
#include "PrimitiveDataTypes.h"

#include "RayLib/Sphere.h"
#include "RayLib/Triangle.h"

#include <nlohmann/json.hpp>

class SurfaceDataLoader : public SurfaceDataLoaderI
{
	private:
	protected:
		const nlohmann::json		node;
		double						time;

	public:
		// Constructor & Destructor
								SurfaceDataLoader(const nlohmann::json& node, double time = 0.0);
								~SurfaceDataLoader() = default;

		const uint32_t			SurfaceDataId() const override;
};

SurfaceDataLoader::SurfaceDataLoader(const nlohmann::json& node, double time)
	: node(node)
	, time(time)
{}

const uint32_t SurfaceDataLoader::SurfaceDataId() const
{
	return SceneIO::LoadNumber<const uint32_t>(node[SceneIO::ID]);
}

class InNodeTriLoader : public SurfaceDataLoader
{
	private:
	protected:
	public:
		// Constructors & Destructor
								InNodeTriLoader(const nlohmann::json& node, double time = 0.0);
								~InNodeTriLoader() = default;

		// Size Determination
		size_t					PrimitiveCount() const override;
		size_t					PrimitiveDataSize(const std::string& primitiveDataType) const override;
		AABB3					PrimitiveAABB() const override;

		// Load Functionality
		const char*				SufaceDataFileExt() const override;

		//
		SceneError				LoadPrimitiveData(float*, const std::string& primitiveDataType)	override;
		SceneError				LoadPrimitiveData(int*, const std::string& primitiveDataType) override;
		SceneError				LoadPrimitiveData(unsigned int*, const std::string& primitiveDataType) override;
};

size_t InNodeTriLoader::PrimitiveCount() const
{
	return 1;
}

size_t InNodeTriLoader::PrimitiveDataSize(const std::string& primitiveDataType) const
{
	if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::POSITION)] ||
	   primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::NORMAL)])
	{
		return sizeof(float) * 3;
	}
	else if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::UV)])
	{
		return sizeof(float) * 2;
	}
	else throw SceneException(SceneError::SURFACE_DATA_TYPE_NOT_FOUND);
}

AABB3 InNodeTriLoader::PrimitiveAABB() const
{
	int posIndex = static_cast<int>(PrimitiveDataType::POSITION);
	const std::string positionName = PrimitiveDataTypeNames[posIndex];
	std::array<Vector3, 3> data =
	{
		SceneIO::LoadVector<3, float>(node[positionName][0], time),
		SceneIO::LoadVector<3, float>(node[positionName][1], time),
		SceneIO::LoadVector<3, float>(node[positionName][2], time)
	};
	return Triangle::BoundingBox(data[0], data[1], data[2]);
}

const char* InNodeTriLoader::SufaceDataFileExt() const
{
	return "";
}

InNodeTriLoader::InNodeTriLoader(const nlohmann::json& node, double time)
	: SurfaceDataLoader(node, time)
{}

SceneError InNodeTriLoader::LoadPrimitiveData(float* dataOut, const std::string& primitiveDataType)
{
	if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::POSITION)] ||
	   primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::NORMAL)])
	{
		std::array<Vector3, 3> data =
		{
			SceneIO::LoadVector<3, float>(node[primitiveDataType][0], time),
			SceneIO::LoadVector<3, float>(node[primitiveDataType][1], time),
			SceneIO::LoadVector<3, float>(node[primitiveDataType][2], time)
		};
		for(int i = 0; i < 9; i++)
		{
			dataOut[i] = data[i / 3][i % 3];
		}
		return SceneError::OK;
	}
	else if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::UV)])
	{
		std::array<Vector2, 3> data =
		{
			SceneIO::LoadVector<2, float>(node[primitiveDataType][0], time),
			SceneIO::LoadVector<2, float>(node[primitiveDataType][1], time),
			SceneIO::LoadVector<2, float>(node[primitiveDataType][2], time)
		};
		for(int i = 0; i < 6; i++)
		{
			dataOut[i] = data[i / 2][i % 2];
		}
		return SceneError::OK;
	}
	else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

SceneError InNodeTriLoader::LoadPrimitiveData(int*, const std::string& primitiveDataType)
{
	if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::POSITION)] ||
	   primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::NORMAL)] ||
	   primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::UV)])
	{
		return SceneError::SURFACE_DATA_INVALID_READ;
	}
	else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

SceneError InNodeTriLoader::LoadPrimitiveData(unsigned int*, const std::string& primitiveDataType)
{
	if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::POSITION)] ||
	   primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::NORMAL)] ||
	   primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::UV)])
	{
		return SceneError::SURFACE_DATA_INVALID_READ;
	}
	else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

class InNodeSphrLoader : public SurfaceDataLoader
{
	private:
	protected:
	public:
		// Constructors & Destructor
								InNodeSphrLoader(const nlohmann::json& node, double time = 0.0);
								~InNodeSphrLoader() = default;

		// Size Determination
		size_t					PrimitiveCount() const override;
		size_t					PrimitiveDataSize(const std::string& primitiveDataType) const override;
		AABB3					PrimitiveAABB() const override;

		// Load Functionality
		const char*				SufaceDataFileExt() const override;

		//
		SceneError				LoadPrimitiveData(float*, const std::string& primitiveDataType)	override;
		SceneError				LoadPrimitiveData(int*, const std::string& primitiveDataType) override;
		SceneError				LoadPrimitiveData(unsigned int*, const std::string& primitiveDataType) override;
};

size_t InNodeSphrLoader::PrimitiveCount() const
{
	return 1;
}

size_t InNodeSphrLoader::PrimitiveDataSize(const std::string& primitiveDataType) const
{
	if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::CENTER)])
	{
		return sizeof(float) * 3;
	}
	else if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::RADIUS)])
	{
		return sizeof(float);
	}
	else throw SceneException(SceneError::SURFACE_DATA_TYPE_NOT_FOUND);
}

AABB3f InNodeSphrLoader::PrimitiveAABB() const
{
	int centerIndex = static_cast<int>(PrimitiveDataType::CENTER);
	int radIndex = static_cast<int>(PrimitiveDataType::RADIUS);

	Vector3 center = SceneIO::LoadVector<3, float>(node[centerIndex], time);
	float radius = SceneIO::LoadNumber<float>(node[radIndex], time);

	return Sphere::BoundingBox(center, radius);
}

const char* InNodeSphrLoader::SufaceDataFileExt() const
{
	return "";
}

InNodeSphrLoader::InNodeSphrLoader(const nlohmann::json& node, double time)
	: SurfaceDataLoader(node, time)
{}

SceneError InNodeSphrLoader::LoadPrimitiveData(float* dataOut, const std::string& primitiveDataType)
{
	if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::POSITION)])
	{
		Vector3 pos = SceneIO::LoadVector<3, float>(node[primitiveDataType], time);
		dataOut[0] = pos[0];
		dataOut[1] = pos[1];
		dataOut[2] = pos[2];
		return SceneError::OK;
	}
	else if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::RADIUS)])
	{
		dataOut[0] = SceneIO::LoadNumber<float>(node[primitiveDataType], time);
		return SceneError::OK;
	}
	else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

SceneError InNodeSphrLoader::LoadPrimitiveData(int*, const std::string& primitiveDataType)
{
	if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::POSITION)] ||
	   primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::RADIUS)])
	{
		return SceneError::SURFACE_DATA_INVALID_READ;
	}
	else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

SceneError InNodeSphrLoader::LoadPrimitiveData(unsigned int*, const std::string& primitiveDataType)
{
	if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::POSITION)] ||
	   primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::RADIUS)])
	{
		return SceneError::SURFACE_DATA_INVALID_READ;
	}
	else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
}

std::unique_ptr<SurfaceDataLoaderI> SurfaceDataIO::GenSurfaceDataLoader(const nlohmann::json& properties, double time)
{
	const std::string ext = SceneIO::StripFileExt(properties[SceneIO::NAME]);

	// There shoudl
	if(ext == NodeSphereName)
	{
		SurfaceDataLoaderI* loader = new InNodeSphrLoader(properties, time);
		return std::unique_ptr<SurfaceDataLoaderI>(loader);
	}
	else if(ext == NodeTriangleName)
	{
		SurfaceDataLoaderI* loader = new InNodeTriLoader(properties, time);
		return std::unique_ptr<SurfaceDataLoaderI>(loader);
	}
	else throw SceneException(SceneError::NO_LOGIC_FOR_SURFACE_DATA);
}