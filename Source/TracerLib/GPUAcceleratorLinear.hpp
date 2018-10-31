template <class PGroup>
GPUAccLinearGroup<PGroup>::GPUAccLinearGroup(const GPUPrimitiveGroupI& pGroup,
											 const TransformStruct* dInvTransforms)
	: GPUAcceleratorGroup<PGroup>(pGroup, dInvTransforms)
	, dAccRanges(nullptr)
	, dLeafList(nullptr)
{}

template <class PGroup>
const char* GPUAccLinearGroup<PGroup>::Type() const
{
	return TypeName.c_str();
}

template <class PGroup>
SceneError GPUAccLinearGroup<PGroup>::InitializeGroup(std::map<uint32_t, AABB3>& aabbOut,
												 // Map of hit keys for all materials
												 // w.r.t matId and primitive type
													  const std::map<TypeIdPair, HitKey>& allHitKeys,
													  // List of surface/material
													  // pairings that uses this accelerator type
													  // and primitive type
													  const std::map<uint32_t, IdPairings>& pairingList,
													  double time)
{
	std::vector<Vector2ul> acceleratorRanges;

	// Iterate over pairings
	size_t totalSize = 0;
	for(const auto& pairings : pairingList)
	{
		PrimitiveRangeList primRangeList;
		HitKeyList hitKeyList;
		primRangeList.fill(Zero2ul);
		hitKeyList.fill(HitKey::InvalidKey);
		
		Vector2ul range = Vector2ul(totalSize, 0);
		
		uint32_t i = 0;
		size_t localSize = 0;
		AABB3 combinedAABB = CoveringAABB3;
		const IdPairings& pList = pairings.second;
		for(const auto& p : pList)
		{
			if(p.first == std::numeric_limits<uint32_t>::max()) break;

			// Union of AABBs
			AABB3 aabb = primitiveGroup.PrimitiveBatchAABB(p.first);
			combinedAABB = combinedAABB.Union(aabb);
			primRangeList[i] = primitiveGroup.PrimitiveBatchRange(p.first);
			hitKeyList[i] = allHitKeys.at(std::make_pair(primitiveGroup.Type(), p.second));
			i++;
		}
		range[1] = localSize;
		totalSize += localSize;
		
		// Put generated AABB
		aabbOut.emplace(pairings.first, combinedAABB);
		primitiveRanges.push_back(primRangeList);
		primitiveMaterialKeys.push_back(hitKeyList);
		acceleratorRanges.push_back(range);
	}
	assert(aabbOut.size() == primitiveRanges.size());
	assert(primitiveRanges.size() == primitiveMaterialKeys.size());
	assert(primitiveMaterialKeys.size() == idLookup.size());
	assert(idLookup.size() == acceleratorRanges.size());

	// Allocate memory
	size_t leafDataSize = totalSize * sizeof(LeafData);
	size_t accRangeSize = idLookup.size() * sizeof(Vector2ul);
	memory = std::move(DeviceMemory(leafDataSize + accRangeSize));
	dLeafList = static_cast<LeafData*>(memory);
	dAccRanges = reinterpret_cast<Vector2ul*>(static_cast<uint8_t*>(memory) + leafDataSize);

	// Copy Leaf counts to cpu memory
	CUDA_CHECK(cudaMemcpy(dAccRanges, acceleratorRanges.data(), accRangeSize,
						  cudaMemcpyHostToDevice));
	return SceneError::OK;
}

template <class PGroup>
SceneError GPUAccLinearGroup<PGroup>::ChangeTime(std::map<uint32_t, AABB3>& aabbOut,
											     // Map of hit keys for all materials
											     // w.r.t matId and primitive type
												 const std::map<TypeIdPair, HitKey>&,
												 // List of surface/material
												 // pairings that uses this accelerator type
												 // and primitive type
												 const std::map<uint32_t, IdPairings>& pairingList,
												 double time)
{
	// TODO:
	return SceneError::OK;
}

template <class PGroup>
uint32_t GPUAccLinearGroup<PGroup>::InnerId(uint32_t surfaceId) const
{
	return idLookup.at(surfaceId);
}

template <class PGroup>
void GPUAccLinearGroup<PGroup>::ConstructAccelerator(uint32_t surface)
{
	using PrimitiveData = typename PGroup::PrimitiveData;
	const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);

	const uint32_t index = idLookup[surface];
	//const Vector2ul& accelRange = acceleratorRanges[index];
	const PrimitiveRangeList& rangeList = primitiveRanges[index];
	const HitKeyList& hitList = primitiveMaterialKeys[index];

	// TODO: check this array copy works
	// KC
	KCConstructLinear<PGroup><<<1,1>>>(// O
									   dLeafList,
									   // Input
									   dAccRanges,
									   hitList.data(),
									   rangeList.data(),
									   primData,
									   index);
	CUDA_KERNEL_CHECK();

}

template <class PGroup>
void GPUAccLinearGroup<PGroup>::ConstructAccelerators(const std::vector<uint32_t>& surfaces)
{
	// TODO: make this a single KC
	for(const uint32_t& id : surfaces)
	{
		ConstructAccelerator(id);
	}
}

template <class PGroup>
void GPUAccLinearGroup<PGroup>::DestroyAccelerator(uint32_t surface)
{
	//...
}

template <class PGroup>
void GPUAccLinearGroup<PGroup>::DestroyAccelerators(const std::vector<uint32_t>& surfaces)
{
	//...
}

template <class PGroup>
size_t GPUAccLinearGroup<PGroup>::UsedGPUMemory() const
{
	return memory.Size();
}

template <class PGroup>
size_t GPUAccLinearGroup<PGroup>::UsedCPUMemory() const
{
	// TODO:
	// Write allocator wrapper for which keeps track of total bytes allocated
	// and deallocated
	return 0;
}

template <class PGroup>
GPUAccLinearBatch<PGroup>::GPUAccLinearBatch(const GPUAcceleratorGroupI& a,
											 const GPUPrimitiveGroupI& p)
	: GPUAcceleratorBatch(a, p)
{}

template <class PGroup>
const char* GPUAccLinearBatch<PGroup>::Type() const
{
	return TypeName.c_str();
}

template <class PGroup>
void GPUAccLinearBatch<PGroup>::Hit(// O
									HitKey* dMaterialKeys,
									PrimitiveId* dPrimitiveIds,
									HitStructPtr dHitStructs,
									// I-O													
									RayGMem* dRays,
									// Input
									const TransformId* dTransformIds,
									const RayId* dRayIds,
									const HitKey* dAcceleratorKeys,
									const uint32_t rayCount) const
{
	// TODO: Is there a better way to implement this
	using PrimitiveData = typename PGroup::PrimitiveData;
	const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);

	KCIntersectLinear<PGroup><<<1, 1>>>
	(
		// O
		dMaterialKeys,
		dPrimitiveIds,
		dHitStructs,
		// I-O
		dRays,
		// Input
		dTransformIds,
		dRayIds,
		dAcceleratorKeys,
		rayCount,
		// Constants
		acceleratorGroup.dLeafList,
		acceleratorGroup.dAccRanges,
		acceleratorGroup.dInverseTransforms,
		//
		primData
	);
	CUDA_KERNEL_CHECK();
}