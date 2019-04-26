#include "GPUPrimitiveTriangle.h"
#include "SceneFileNode.h"

#include "RayLib/PrimitiveDataTypes.h"
#include "RayLib/SurfaceDataIO.h"
#include "RayLib/SceneError.h"
#include "RayLib/Log.h"

// Generics
GPUPrimitiveTriangle::GPUPrimitiveTriangle()
	: totalPrimitiveCount(0)
{}

const char* GPUPrimitiveTriangle::Type() const
{	
	return TypeName;
}

SceneError GPUPrimitiveTriangle::InitializeGroup(const std::set<SceneFileNode>& surfaceDataNodes, 
												 double time)
{
	// Generate Loaders
	std::vector<std::unique_ptr<SurfaceDataLoaderI>> loaders;
	for(const SceneFileNode& s : surfaceDataNodes)
	{
		//const nlohmann::json& node = static_cast<const nlohmann::json&>(s);
		loaders.push_back(std::move(SurfaceDataIO::GenSurfaceDataLoader(s, time)));
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
		batchAABBs.emplace(surfId, loader->PrimitiveAABB());
	}

	const size_t totalVertexCount = totalPrimitiveCount * 3;
	std::vector<float> postitionsCPU(totalVertexCount * 3);
	std::vector<float> normalsCPU(totalVertexCount * 3);
	std::vector<float> uvsCPU(totalVertexCount * 2);
	size_t offset = 0;
	for(const auto& loader : loaders)
	{
		if(e != loader->LoadPrimitiveData(postitionsCPU.data() + offset,
											PrimitiveDataTypeToString(PrimitiveDataType::POSITION)))
			return e;
		if(e != loader->LoadPrimitiveData(normalsCPU.data() + offset,
											PrimitiveDataTypeToString(PrimitiveDataType::NORMAL)))
			return e;
		if(e != loader->LoadPrimitiveData(uvsCPU.data() + offset,
											PrimitiveDataTypeToString(PrimitiveDataType::UV)))
			return e;

		offset += loader->PrimitiveCount();
	}
	assert(offset == totalPrimitiveCount);

	// All loaded to CPU
	// Now copy to GPU
	// Alloc
	memory = std::move(DeviceMemory(sizeof(Vector4f) * 2 * totalVertexCount));
	float* dPositionsU = static_cast<float*>(memory);
	float* dNormalsV = static_cast<float*>(memory) + totalVertexCount * 4;

	CUDA_CHECK(cudaMemcpy2D(dPositionsU, sizeof(Vector4f),
							postitionsCPU.data(), sizeof(float) * 3,
							sizeof(float) * 3, totalVertexCount,
							cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy2D(dNormalsV, sizeof(Vector4f),
							normalsCPU.data(), sizeof(float) * 3,
							sizeof(float) * 3, totalVertexCount,
							cudaMemcpyHostToDevice));
	// Strided Copy of UVs
	CUDA_CHECK(cudaMemcpy2D(dPositionsU + 3, sizeof(Vector4f),
							uvsCPU.data(), sizeof(float) * 2,
							sizeof(float), totalVertexCount,
							cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy2D(dNormalsV + 3, sizeof(Vector4f),
							uvsCPU.data() + 1, sizeof(float) * 2,
							sizeof(float), totalVertexCount,
							cudaMemcpyHostToDevice));

	// Set Main Pointers of batch
	dData.positionsU = reinterpret_cast<Vector4f*>(dPositionsU);
	dData.normalsV = reinterpret_cast<Vector4f*>(dNormalsV);
	return e;
}

SceneError GPUPrimitiveTriangle::ChangeTime(const std::set<SceneFileNode>& surfaceDatalNodes, double time)
{
	// Generate Loaders
	std::vector<std::unique_ptr<SurfaceDataLoaderI>> loaders;
	for(const SceneFileNode& s : surfaceDatalNodes)
	{
		loaders.push_back(std::move(SurfaceDataIO::GenSurfaceDataLoader(s, time)));
	}

	SceneError e = SceneError::OK;
	std::vector<float> postitionsCPU, normalsCPU, uvsCPU;
	for(const auto& loader : loaders)
	{
		uint32_t surfId = loader->SurfaceDataId();
		Vector2ul range = batchRanges.at(loader->SurfaceDataId());
		size_t primitiveCount = loader->PrimitiveCount();
		assert((range[1] - range[0]) == primitiveCount);

		batchAABBs.at(surfId) = loader->PrimitiveAABB();
		// Alloc enough CPU space
		postitionsCPU.resize(primitiveCount * 3);
		normalsCPU.resize(primitiveCount * 2);
		uvsCPU.resize(primitiveCount * 2);

		if(e != loader->LoadPrimitiveData(postitionsCPU.data(),
										  PrimitiveDataTypeToString(PrimitiveDataType::POSITION)))
			return e;
		if(e != loader->LoadPrimitiveData(normalsCPU.data(),
										  PrimitiveDataTypeToString(PrimitiveDataType::NORMAL)))
			return e;
		if(e != loader->LoadPrimitiveData(uvsCPU.data(), 
										  PrimitiveDataTypeToString(PrimitiveDataType::UV)))
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

		// Wait on Default Stream
		CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)0));
	}
	return e;
}

Vector2ul GPUPrimitiveTriangle::PrimitiveBatchRange(uint32_t surfaceDataId) const
{
	return batchRanges.at(surfaceDataId);
}

AABB3 GPUPrimitiveTriangle::PrimitiveBatchAABB(uint32_t surfaceDataId) const
{
	return batchAABBs.at(surfaceDataId);
}

bool GPUPrimitiveTriangle::CanGenerateData(const std::string& s) const
{
	return (s == PrimitiveDataTypeToString(PrimitiveDataType::POSITION) ||
			s == PrimitiveDataTypeToString(PrimitiveDataType::NORMAL) ||
			s == PrimitiveDataTypeToString(PrimitiveDataType::UV));
}