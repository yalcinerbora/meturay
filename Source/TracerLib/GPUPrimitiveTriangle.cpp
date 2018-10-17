#include "GPUPrimitiveTriangle.h"
#include "PrimitiveDataTypes.h"

#include "RayLib/SurfaceDataIO.h"
#include "RayLib/SceneError.h"

// Generics
GPUPrimitiveTriangle::GPUPrimitiveTriangle()
	: data{ nullptr, nullptr}
{}

const std::string& GPUPrimitiveTriangle::PrimitiveType() const
{
	return "Triangle";
}

SceneError GPUPrimitiveTriangle::InitializeGroup(const std::vector<SceneFileNode>& surfaceDatalNodes, 
												 double time)
{
	// Generate Loaders
	std::vector<std::unique_ptr<SurfaceDataLoaderI>> loaders;
	for(const SceneFileNode& s : surfaceDatalNodes)
	{
		loaders.push_back(std::move(SurfaceDataIO::GenSurfaceDataLoader(s)));		
	}

	SceneError e = SceneError::OK;
	size_t totalPrimCount = 0;
	for(const auto& loader : loaders)
	{
		totalPrimCount = loader->PrimitiveCount();
	}

	std::vector<float> postitionsCPU(totalPrimCount * 3);
	std::vector<float> normalsCPU(totalPrimCount * 3);
	std::vector<float> uvsCPU(totalPrimCount * 2);

	size_t offset = 0;
	for(const auto& loader : loaders)
	{
		if(e != loader->LoadPrimitiveData(postitionsCPU.data() + offset,
										  PrimBasicDataTypeToString(PrimitiveBasicDataType::POSITION)))
			return e;
		if(e != loader->LoadPrimitiveData(normalsCPU.data() + offset,
										  PrimBasicDataTypeToString(PrimitiveBasicDataType::NORMAL)))
			return e;
		if(e != loader->LoadPrimitiveData(uvsCPU.data() + offset,
										  PrimBasicDataTypeToString(PrimitiveBasicDataType::UV)))
			return e;
		
		offset += loader->PrimitiveCount();
	}
	assert(offset == totalPrimCount);


	// All loaded to CPU now copy to GPU

	CUDA_CHECK(cudaMemcpy2D());
	CUDA_CHECK(cudaMemcpy2D());
	CUDA_CHECK(cudaMemcpy2D());
}

SceneError GPUPrimitiveTriangle::ChangeTime(const const std::vector<SceneFileNode>& surfaceDatalNodes, double time)
{
	//
	for(const SceneFileNode& s : surfaceDatalNodes)
	{
		std::vector<Vector3f> postitionsCPU;
		std::vector<Vector3f> normalsCPU;
		std::vector<Vector2f> uvsCPU;

		// Copy partially
	}
}

Vector2ui GPUPrimitiveTriangle::PrimitiveBatchRange(uint32_t surfaceDataId)
{
	return batchRanges[surfaceDataId];
}

bool GPUPrimitiveTriangle::CanGenerateData(const std::string& s) const
{
	return (s == PrimBasicDataTypeToString(PrimitiveBasicDataType::POSITION) ||
			s == PrimBasicDataTypeToString(PrimitiveBasicDataType::NORMAL) ||
			s == PrimBasicDataTypeToString(PrimitiveBasicDataType::UV));
}