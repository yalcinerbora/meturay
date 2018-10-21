#include "LinearAcceleratorKernels.cuh"

template <class P>
GPUAccLinearGroup<P>::GPUAccLinearGroup(const P& pGroup,
										const TransformStruct* dInvTransforms)
	: dInverseTransforms(dInvTransforms)
	, primitiveGroup(pGroup)
	, dLeafCounts(nullptr)
	, dLeafList(nullptr)
{}

template <class P>
const std::string& GPUAccLinearGroup<P>::Type() const
{
	return TypeName;
}

template <class P>
SceneError GPUAccLinearGroup<P>::InitializeGroup(const std::map<uint32_t, HitKey>& materialKeyList,
													// List of surface nodes
													// that uses this accelerator type
													// w.r.t. this prim group
													const std::vector<SceneFileNode>&)
{
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

template<class A, class P>
GPUAccLinearBatch<A, P>::GPUAccLinearBatch(GPUAcceleratorGroupI& a,
										   GPUPrimitiveGroupI& p)
	: acceleratorGroup(static_cast<A*>(a))
	, primitiveGroup(static_cast<P*>(p))
{}

template<class A, class P>
const std::string& GPUAccLinearBatch<A, P>::Type() const
{
	return TypeName;
}

template<class A, class P>
void GPUAccLinearBatch<A, P>::Hit(// O
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
	using PrimitiveData = typename P::PrimitiveData;
	PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);

	//TODO:.....
	KCIntersectLinear<P><<<1,1>>>
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
		dLeafList,
		dLeafCounts,
		dInverseTransforms,
		//								   								   
		primData
	);
}

template <class A, class P>
const GPUPrimitiveGroupI& GPUAccLinearBatch<A, P>::PrimitiveGroup() const
{
	return primitiveGroup;
}

template <class A, class P>
const GPUAcceleratorGroupI& GPUAccLinearBatch<A, P>::AcceleratorGroup() const
{
	return acceleratorGroup;
}