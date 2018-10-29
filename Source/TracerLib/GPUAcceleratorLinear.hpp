template <class P>
GPUAccLinearGroup<P>::GPUAccLinearGroup(const GPUPrimitiveGroupI& pGroup,
										const TransformStruct* dInvTransforms)
	: GPUAcceleratorGroup<P>(pGroup, dInvTransforms)
	, dLeafCounts(nullptr)
	, dLeafList(nullptr)
{}

template <class P>
const char* GPUAccLinearGroup<P>::Type() const
{
	return TypeName.c_str();
}

template <class P>
SceneError GPUAccLinearGroup<P>::InitializeGroup(const std::map<uint32_t, HitKey>& materialKeyList,												 
												 const std::vector<SceneFileNode>& nodeList,
												 double time)
{
	for(const SceneFileNode& s : nodeList)
	{
		MaterialList l = NodeDataRead::LinearAcceleator(s, time);
		PrimList l = 
	}

	//....
	return SceneError::OK;
}

template <class P>
void GPUAccLinearGroup<P>::ConstructAccelerator(uint32_t surface)
{
	//...
}

template <class P>
void GPUAccLinearGroup<P>::ConstructAccelerators(const std::vector<uint32_t>& surfaces)
{
	//...
}

template <class P>
void GPUAccLinearGroup<P>::DestroyAccelerator(uint32_t surface)
{
	//...
}

template <class P>
void GPUAccLinearGroup<P>::DestroyAccelerators(const std::vector<uint32_t>& surfaces)
{
	//...
}

template <class P>
size_t GPUAccLinearGroup<P>::UsedGPUMemory() const
{
	return memory.Size();
}

template <class P>
size_t GPUAccLinearGroup<P>::UsedCPUMemory() const
{
	// TODO:
	// Write allocator wrapper for which keeps track of total bytes allocated
	// and deallocated
	return 0;
}

template <class P>
GPUAccLinearBatch<P>::GPUAccLinearBatch(const GPUAcceleratorGroupI& a,
										const GPUPrimitiveGroupI& p)
	: GPUAcceleratorBatch(a, p)
{}

template <class P>
const char* GPUAccLinearBatch<P>::Type() const
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
		acceleratorGroup.dLeafCounts,
		acceleratorGroup.dInverseTransforms,
		//								   								   
		primData
	);
}