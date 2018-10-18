#include "GPUPrimitiveTriangle.h"
#include "PrimitiveDataTypes.h"

#include "RayLib/SurfaceDataIO.h"
#include "RayLib/SceneError.h"

// Generics
GPUPrimitiveTriangle::GPUPrimitiveTriangle()
	: dData{ nullptr, nullptr}
	, totalPrimitiveCount(0)
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
	totalPrimitiveCount = 0;
	for(const auto& loader : loaders)
	{
		uint32_t surfId = loader->SurfaceDataId();
		uint64_t start = totalPrimitiveCount;
		uint64_t end = start + loader->PrimitiveCount();
		totalPrimitiveCount = end;

		batchRanges.emplace(surfId, Vector2ul(start, end));
	}

	std::vector<float> postitionsCPU(totalPrimitiveCount * 3);
	std::vector<float> normalsCPU(totalPrimitiveCount * 3);
	std::vector<float> uvsCPU(totalPrimitiveCount * 2);
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
	assert(offset == totalPrimitiveCount);

	// All loaded to CPU
	// Now copy to GPU
	// Alloc
	memory = std::move(DeviceMemory(sizeof(Vector4f) * 2 * totalPrimitiveCount));
	float* dPositionsU = static_cast<float*>(memory);
	float* dNormalsV = static_cast<float*>(memory) + totalPrimitiveCount;

	CUDA_CHECK(cudaMemcpy2D(dPositionsU, sizeof(Vector4f),
							postitionsCPU.data(), sizeof(float) * 3,
							sizeof(float) * 3, totalPrimitiveCount,
							cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy2D(dNormalsV, sizeof(Vector4f),
							 normalsCPU.data(), sizeof(float) * 3,
							 sizeof(float) * 3, totalPrimitiveCount,
							 cudaMemcpyHostToDevice));
	// Strided Copy of UVs
	CUDA_CHECK(cudaMemcpy2D(dPositionsU + 3, sizeof(Vector4f),
							uvsCPU.data(), sizeof(float) * 2,
							sizeof(float), totalPrimitiveCount,
							cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy2D(dNormalsV + 3, sizeof(Vector4f),
							uvsCPU.data() + 1, sizeof(float) * 2,
							sizeof(float), totalPrimitiveCount,
							cudaMemcpyHostToDevice));

	// Set Main Pointers of batch
	dData.positionsU = reinterpret_cast<Vector4f*>(dPositionsU);
	dData.normalsV = reinterpret_cast<Vector4f*>(dNormalsV);
}

SceneError GPUPrimitiveTriangle::ChangeTime(const const std::vector<SceneFileNode>& surfaceDatalNodes, double time)
{
	// Generate Loaders
	std::vector<std::unique_ptr<SurfaceDataLoaderI>> loaders;
	for(const SceneFileNode& s : surfaceDatalNodes)
	{
		loaders.push_back(std::move(SurfaceDataIO::GenSurfaceDataLoader(s)));
	}

	SceneError e = SceneError::OK;
	for(const auto& loader : loaders)
	{
		Vector2ui range = batchRanges[loader->SurfaceDataId()];
		size_t primitiveCount = loader->PrimitiveCount();
		assert((range[1] - range[0]) == primitiveCount);

		std::vector<float> postitionsCPU(primitiveCount * 3);
		std::vector<float> normalsCPU(primitiveCount * 2);
		std::vector<float> uvsCPU(primitiveCount * 2);

		if(e != loader->LoadPrimitiveData(postitionsCPU.data(),
										  PrimBasicDataTypeToString(PrimitiveBasicDataType::POSITION)))
			return e;
		if(e != loader->LoadPrimitiveData(normalsCPU.data(),
										  PrimBasicDataTypeToString(PrimitiveBasicDataType::NORMAL)))
			return e;
		if(e != loader->LoadPrimitiveData(uvsCPU.data(), PrimBasicDataTypeToString(PrimitiveBasicDataType::UV)))
			return e;
		
		// Copy
		float* dPositionsU = static_cast<float*>(memory);
		float* dNormalsV = static_cast<float*>(memory) + totalPrimitiveCount;
		
		// Pos and Normal
		CUDA_CHECK(cudaMemcpy2D(dPositionsU + range[0], sizeof(Vector4f),
								postitionsCPU.data(), sizeof(float) * 3,
								sizeof(float) * 3, primitiveCount,
								cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy2D(dNormalsV + range[0], sizeof(Vector4f),
								normalsCPU.data(), sizeof(float) * 3,
								sizeof(float) * 3, primitiveCount,
								cudaMemcpyHostToDevice));
		// Strided Copy of UVs
		CUDA_CHECK(cudaMemcpy2D(dPositionsU + range[0] + 3, sizeof(Vector4f),
								uvsCPU.data(), sizeof(float) * 2,
								sizeof(float), primitiveCount,
								cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy2D(dNormalsV + range[0] + 3, sizeof(Vector4f),
								uvsCPU.data() + 1, sizeof(float) * 2,
								sizeof(float), primitiveCount,
								cudaMemcpyHostToDevice));
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