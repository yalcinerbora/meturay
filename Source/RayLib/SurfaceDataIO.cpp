#include "SurfaceDataIO.h"
#include "SceneIO.h"
#include "SceneFileNode.h"
#include "PrimitiveDataTypes.h"

class SurfaceDataLoader : public SurfaceDataLoaderI
{
	private:		
	protected:
		SceneFileNode			node;
		double					time;

	public:
		// Constructor & Destructor
								SurfaceDataLoader(const SceneFileNode& node, double time = 0.0);
								~SurfaceDataLoader() = default;

		const uint32_t			SurfaceDataId() const override;
};

SurfaceDataLoader::SurfaceDataLoader(const SceneFileNode& node, double time)
	: node(node)
	, time(time)
{}

const uint32_t SurfaceDataLoader::SurfaceDataId() const
{
	return SceneIO::LoadNumber<const uint32_t>(node.jsn[SceneIO::ID]); 
}

class InNodeTriLoader : public SurfaceDataLoader
{
	private:
	protected:
	public:
		// Constructors & Destructor
								InNodeTriLoader(const SceneFileNode& node, double time = 0.0);
								~InNodeTriLoader() = default;

		// Size Determination
		size_t					PrimitiveCount() const override;
		size_t					PrimitiveDataSize(const std::string& primitiveDataType) const override;

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

const char* InNodeTriLoader::SufaceDataFileExt() const
{
	return "";
}

InNodeTriLoader::InNodeTriLoader(const SceneFileNode& node, double time)
	: SurfaceDataLoader(node, time)
{}

SceneError InNodeTriLoader::LoadPrimitiveData(float* dataOut, const std::string& primitiveDataType)
{
	if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::POSITION)] ||
	   primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::NORMAL)])
	{
		Vector3 data = SceneIO::LoadVector<3, float>(node.jsn[primitiveDataType], time);
		dataOut[0] = data[0];
		dataOut[1] = data[1];
		dataOut[2] = data[2];
		return SceneError::OK;
	}
	else if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::UV)])
	{		
		Vector2 uv = SceneIO::LoadVector<2, float>(node.jsn[primitiveDataType], time);
		dataOut[0] = uv[0];
		dataOut[1] = uv[1];
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
								InNodeSphrLoader(const SceneFileNode& node, double time = 0.0);
								~InNodeSphrLoader() = default;

		// Size Determination
		size_t					PrimitiveCount() const override;
		size_t					PrimitiveDataSize(const std::string& primitiveDataType) const override;

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
	if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::POSITION)])
	{
		return sizeof(float) * 3;
	}
	else if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::RADIUS)])
	{
		return sizeof(float);
	}
	else throw SceneException(SceneError::SURFACE_DATA_TYPE_NOT_FOUND);
}

const char* InNodeSphrLoader::SufaceDataFileExt() const
{
	return "";
}

InNodeSphrLoader::InNodeSphrLoader(const SceneFileNode& node, double time)
	: SurfaceDataLoader(node, time)
{}

SceneError InNodeSphrLoader::LoadPrimitiveData(float* dataOut, const std::string& primitiveDataType)
{
	if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::POSITION)])
	{
		Vector3 pos = SceneIO::LoadVector<3, float>(node.jsn[primitiveDataType], time);
		dataOut[0] = pos[0];
		dataOut[1] = pos[1];
		dataOut[2] = pos[2];
		return SceneError::OK;
	}
	else if(primitiveDataType == PrimitiveDataTypeNames[static_cast<int>(PrimitiveDataType::RADIUS)])
	{		
		dataOut[0] = SceneIO::LoadNumber<float>(node.jsn[primitiveDataType], time);
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

std::unique_ptr<SurfaceDataLoaderI> SurfaceDataIO::GenSurfaceDataLoader(const SceneFileNode& properties, double time)
{
	const std::string ext = SceneIO::StripFileExt(properties.jsn[SceneIO::NAME]);

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