#include "GPUPrimitiveSphere.h"

#include "RayLib/PrimitiveDataTypes.h"
#include "RayLib/SurfaceDataIO.h"
#include "RayLib/SceneError.h"
#include "RayLib/SceneFileNode.h"

// Generics
GPUPrimitiveSphere::GPUPrimitiveSphere()
	: totalPrimitiveCount(0)
{}

const char* GPUPrimitiveSphere::Type() const
{
	return TypeName;
}

SceneError GPUPrimitiveSphere::InitializeGroup(const std::set<SceneFileNode>& surfaceDatalNodes,
												 double time)
{
	// Generate Loaders
	std::vector<std::unique_ptr<SurfaceDataLoaderI>> loaders;
	for(const SceneFileNode& s : surfaceDatalNodes)
	{
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

	std::vector<float> postitionsCPU(totalPrimitiveCount * 3);
	std::vector<float> radiusCPU(totalPrimitiveCount);
	size_t offset = 0;
	for(const auto& loader : loaders)
	{
		if(e != loader->LoadPrimitiveData(postitionsCPU.data() + offset,
										  PrimitiveDataTypeToString(PrimitiveDataType::POSITION)))
			return e;
		if(e != loader->LoadPrimitiveData(radiusCPU.data() + offset,
										  PrimitiveDataTypeToString(PrimitiveDataType::RADIUS)))
			return e;
		offset += loader->PrimitiveCount();
	}
	assert(offset == totalPrimitiveCount);

	// All loaded to CPU
	// Now copy to GPU
	// Alloc
	memory = std::move(DeviceMemory(sizeof(Vector4f) * totalPrimitiveCount));
	float* dCentersRadius = static_cast<float*>(memory);

	CUDA_CHECK(cudaMemcpy2D(dCentersRadius, sizeof(Vector4f),
							postitionsCPU.data(), sizeof(float) * 3,
							sizeof(float) * 3, totalPrimitiveCount,
							cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy2D(dCentersRadius + 3, sizeof(Vector4f),
							radiusCPU.data(), sizeof(float),
							sizeof(float), totalPrimitiveCount,
							cudaMemcpyHostToDevice));

	// Set Main Pointers of batch
	dData.centerRadius = reinterpret_cast<Vector4f*>(dCentersRadius);
	return e;
}

SceneError GPUPrimitiveSphere::ChangeTime(const std::set<SceneFileNode>& surfaceDatalNodes, double time)
{
	// Generate Loaders
	std::vector<std::unique_ptr<SurfaceDataLoaderI>> loaders;
	for(const SceneFileNode& s : surfaceDatalNodes)
	{
		loaders.push_back(std::move(SurfaceDataIO::GenSurfaceDataLoader(s, time)));
	}

	SceneError e = SceneError::OK;
	std::vector<float> postitionsCPU, radiusCPU;
	for(const auto& loader : loaders)
	{
		Vector2ul range = batchRanges[loader->SurfaceDataId()];
		size_t primitiveCount = loader->PrimitiveCount();
		assert((range[1] - range[0]) == primitiveCount);

		postitionsCPU.resize(primitiveCount * 3);
		radiusCPU.resize(primitiveCount);
	
		if(e != loader->LoadPrimitiveData(postitionsCPU.data(),
										  PrimitiveDataTypeToString(PrimitiveDataType::POSITION)))
			return e;
		if(e != loader->LoadPrimitiveData(radiusCPU.data(),
										  PrimitiveDataTypeToString(PrimitiveDataType::RADIUS)))
			return e;

		// Copy
		float* dCentersRadius = static_cast<float*>(memory);
		CUDA_CHECK(cudaMemcpy2D(dCentersRadius, sizeof(Vector4f),
								postitionsCPU.data(), sizeof(float) * 3,
								sizeof(float) * 3, primitiveCount,
								cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemcpy2D(dCentersRadius + 3, sizeof(Vector4f),
								radiusCPU.data(), sizeof(float),
								sizeof(float), primitiveCount,
								cudaMemcpyHostToDevice));
	}
	return e;
}

Vector2ul GPUPrimitiveSphere::PrimitiveBatchRange(uint32_t surfaceDataId) const
{
	return batchRanges.at(surfaceDataId);
}

AABB3 GPUPrimitiveSphere::PrimitiveBatchAABB(uint32_t surfaceDataId) const
{
	return batchAABBs.at(surfaceDataId);
}

bool GPUPrimitiveSphere::CanGenerateData(const std::string& s) const
{
	return (s == PrimitiveDataTypeToString(PrimitiveDataType::POSITION) ||
			s == PrimitiveDataTypeToString(PrimitiveDataType::NORMAL) ||
			s == PrimitiveDataTypeToString(PrimitiveDataType::UV));
}