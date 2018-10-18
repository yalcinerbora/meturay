
template <class P>
GPUAcceleratorLinear<P>::GPUAcceleratorLinear(const P& pGroup,
											  const TransformStruct* dInvTransforms)
	: dInverseTransforms(dInvTransforms)
	, primitiveGroup(pGroup)
	, dLeafCounts(nullptr)
	, dLeafList(nullptr)
{}

template <class P>
SceneError GPUAcceleratorLinear<P>::InitializeGroup(const std::map<uint32_t, HitKey>& materialKeyList,
													// List of surface nodes
													// that uses this accelerator type
													// w.r.t. this prim group
													const std::vector<SceneFileNode>&)
{
	....
}

template <class P>
void GPUAcceleratorLinear<P>::ConstructAccelerator(uint32_t surface)
{
	...
}

template <class P>
void GPUAcceleratorLinear<P>::ConstructAccelerators(const std::vector<uint32_t>& surfaces)
{
	...
}

template <class P>
void GPUAcceleratorLinear<P>::DestroyAccelerator(uint32_t surface)
{
	...
}

template <class P>
void GPUAcceleratorLinear<P>::DestroyAccelerators(const std::vector<uint32_t>& surfaces)
{
	...
}

template <class P>
size_t GPUAcceleratorLinear<P>::UsedGPUMemory() const
{
	return memory.Size();
}

template <class P>
size_t GPUAcceleratorLinear<P>::UsedCPUMemory() const
{
	// TODO:
	// Write allocator wrapper for which keeps track of total bytes allocated
	// and deallocated
	return 0;
}

template <class P>
void GPUAcceleratorLinear<P>::Hit(// O
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
	using PrimitiveData = typename P::PrimitiveData;

	//TODO:.....
	KCIntersectLinear<P><<<1, 1>>>(// O
								   dMaterialKeys,
								   dPrimitiveIds,
								   dHitStructs,
								   // I-O
								   dRays,
								   // Input
								   dTransformIds,
								   dRayIds,
								   dHitKeys,
								   rayCount,
								   // Constants
								   dLeafList,
								   dLeafCounts,
								   dInverseTransforms,
								   								   
								   // TODO:: how to call primitive data
								   ???
								   primitiveGroup.Data<PrimitiveData>());

}

template <class P>
const GPUPrimitiveGroupI& GPUAcceleratorLinear<P>::PrimitiveGroup() const
{
	return primitiveGroup;
}

template <class P>
const GPUAcceleratorGroupI& GPUAcceleratorLinear<P>::AcceleratorGroup() const
{
	return *this;
}