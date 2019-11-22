#include "TracerLogicGenerator.h"

#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveEmpty.h"
#include "GPUAcceleratorLinear.cuh"

#include "GPUMaterialI.h"

// Type to utilize the generated ones
extern template class GPUAccLinearGroup<GPUPrimitiveTriangle>;
extern template class GPUAccLinearGroup<GPUPrimitiveSphere>;

extern template class GPUAccLinearBatch<GPUPrimitiveTriangle>;
extern template class GPUAccLinearBatch<GPUPrimitiveSphere>;

// Typedefs for ease of read
using GPUAccTriLinearGroup = GPUAccLinearGroup<GPUPrimitiveTriangle>;
using GPUAccSphrLinearGroup = GPUAccLinearGroup<GPUPrimitiveSphere>;

using GPUAccTriLinearBatch = GPUAccLinearBatch<GPUPrimitiveTriangle>;
using GPUAccSphrLinearBatch = GPUAccLinearBatch<GPUPrimitiveSphere>;

// Some Instantiations
// Constructors
template GPUPrimitiveGroupI* TypeGenWrappers::DefaultConstruct<GPUPrimitiveGroupI,
                                                               GPUPrimitiveTriangle>();
template GPUPrimitiveGroupI* TypeGenWrappers::DefaultConstruct<GPUPrimitiveGroupI,
                                                               GPUPrimitiveSphere>();

template GPUAcceleratorGroupI* TypeGenWrappers::AccelGroupConstruct<GPUAcceleratorGroupI,
                                                                    GPUAccTriLinearGroup>(const GPUPrimitiveGroupI&,
                                                                                          const TransformStruct*);
template GPUAcceleratorGroupI* TypeGenWrappers::AccelGroupConstruct<GPUAcceleratorGroupI,
                                                                    GPUAccSphrLinearGroup>(const GPUPrimitiveGroupI&,
                                                                                           const TransformStruct*);

template GPUAcceleratorBatchI* TypeGenWrappers::AccelBatchConstruct<GPUAcceleratorBatchI,
                                                                    GPUAccTriLinearBatch>(const GPUAcceleratorGroupI&,
                                                                                          const GPUPrimitiveGroupI&);
template GPUAcceleratorBatchI* TypeGenWrappers::AccelBatchConstruct<GPUAcceleratorBatchI,
                                                                    GPUAccSphrLinearBatch>(const GPUAcceleratorGroupI&,
                                                                                           const GPUPrimitiveGroupI&);
// Destructors
template void TypeGenWrappers::DefaultDestruct(GPUPrimitiveGroupI*);
template void TypeGenWrappers::DefaultDestruct(GPUMaterialGroupI*);
template void TypeGenWrappers::DefaultDestruct(GPUAcceleratorGroupI*);

template void TypeGenWrappers::DefaultDestruct(GPUAcceleratorBatchI*);
template void TypeGenWrappers::DefaultDestruct(GPUMaterialBatchI*);

uint32_t TracerLogicGenerator::CalculateHitStruct()
{
    uint32_t currentSize = std::numeric_limits<uint32_t>::min();
    for(const auto& primPtr : primGroups)
    {
        uint32_t hitSize = primPtr.second->PrimitiveHitSize();
        currentSize = std::max(hitSize, currentSize);
    }

    // Properly Align
    currentSize = ((currentSize + sizeof(uint32_t) - 1) / sizeof(uint32_t)) * sizeof(uint32_t);
    return currentSize;
}

// Helper Funcs
bool TracerLogicGenerator::FindSharedLib(SharedLib* libOut,
                                         const std::string& libName) const
{
    //....
    return false;
}

TracerLogicGenerator::TracerLogicGenerator()
    : baseAccelerator(nullptr, TypeGenWrappers::DefaultDestruct<GPUBaseAcceleratorI>)
    , tracerPtr(nullptr, nullptr)
{
    using namespace TypeGenWrappers;

    // Primitive Defaults
    primGroupGenerators.emplace(GPUPrimitiveTriangle::TypeName(),
                                GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveTriangle>,
                                                DefaultDestruct<GPUPrimitiveGroupI>));
    primGroupGenerators.emplace(GPUPrimitiveSphere::TypeName(),
                                GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveSphere>,
                                                DefaultDestruct<GPUPrimitiveGroupI>));
    primGroupGenerators.emplace(GPUPrimitiveEmpty::TypeName(),
                                GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveEmpty>,
                                                DefaultDestruct<GPUPrimitiveGroupI>));

    // Accelerator Types
    accelGroupGenerators.emplace(GPUAccTriLinearGroup::TypeName(),
                                 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccTriLinearGroup>,
                                                  DefaultDestruct<GPUAcceleratorGroupI>));
    accelGroupGenerators.emplace(GPUAccSphrLinearGroup::TypeName(),
                                 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccSphrLinearGroup>,
                                                  DefaultDestruct<GPUAcceleratorGroupI>));

    accelBatchGenerators.emplace(GPUAccTriLinearBatch::TypeName(),
                                 GPUAccelBatchGen(AccelBatchConstruct<GPUAcceleratorBatchI, GPUAccTriLinearBatch>,
                                                  DefaultDestruct<GPUAcceleratorBatchI>));
    accelBatchGenerators.emplace(GPUAccSphrLinearBatch::TypeName(),
                                 GPUAccelBatchGen(AccelBatchConstruct<GPUAcceleratorBatchI, GPUAccSphrLinearBatch>,
                                                  DefaultDestruct<GPUAcceleratorBatchI>));

    // Base Accelerator
    baseAccelGenerators.emplace(GPUBaseAcceleratorLinear::TypeName(),
                                GPUBaseAccelGen(DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorLinear>,
                                                DefaultDestruct<GPUBaseAcceleratorI>));

    // Default Types are loaded
    // Other Types are strongly tied to base tracer logic
    // i.e. Auxiliary Struct Etc.
}

SceneError TracerLogicGenerator::GeneratePrimitiveGroup(GPUPrimitiveGroupI*& pg,
                                                        const std::string& primitiveType)
{
    pg = nullptr;
    auto loc = primGroups.find(primitiveType);
    if(loc == primGroups.end())
    {
        // Cannot Find Already Constructed Type
        // Generate
        auto loc = primGroupGenerators.find(primitiveType);
        if(loc == primGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_PRIMITIVE;

        GPUPrimGPtr ptr = loc->second();
        pg = ptr.get();
        primGroups.emplace(primitiveType, std::move(ptr));
    }
    else pg = loc->second.get();
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateAcceleratorGroup(GPUAcceleratorGroupI*& ag,
                                                          const GPUPrimitiveGroupI& pg,
                                                          const TransformStruct* t,
                                                          const std::string& accelType)
{
    ag = nullptr;
    auto loc = accelGroups.find(accelType);
    if(loc == accelGroups.end())
    {
        // Cannot Find Already Constructed Type
        // Generate
        auto loc = accelGroupGenerators.find(accelType);
        if(loc == accelGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_ACCELERATOR;

        GPUAccelGPtr ptr = loc->second(pg, t);
        ag = ptr.get();
        accelGroups.emplace(accelType, std::move(ptr));
    }
    else ag = loc->second.get();
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateAcceleratorBatch(GPUAcceleratorBatchI*& ab,
                                                          const GPUAcceleratorGroupI& ag,
                                                          const GPUPrimitiveGroupI& pg,
                                                          uint32_t keyBatchId)
{
    ab = nullptr;
    // Check duplicate batchId
    if(accelBatchMap.find(keyBatchId) != accelBatchMap.end())
        return SceneError::INTERNAL_DUPLICATE_ACCEL_ID;

    const std::string batchType = std::string(ag.Type());

    auto loc = accelBatches.find(batchType);
    if(loc == accelBatches.end())
    {
        // Cannot Find Already Constructed Type
        // Generate
        auto loc = accelBatchGenerators.find(batchType);
        if(loc == accelBatchGenerators.end()) return SceneError::NO_LOGIC_FOR_ACCELERATOR;

        GPUAccelBPtr ptr = loc->second(ag, pg);
        ab = ptr.get();
        accelBatches.emplace(batchType, std::move(ptr));
        accelBatchMap.emplace(keyBatchId, ab);
    }
    else ab = loc->second.get();
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateMaterialGroup(GPUMaterialGroupI*& mg,
                                                       const std::string& materialType,
                                                       const int gpuId)
{
    mg = nullptr;
    auto loc = matGroups.find(std::make_pair(materialType, gpuId));
    if(loc == matGroups.end())
    {
        // Cannot Find Already Constructed Type
        // Generate
        auto loc = matGroupGenerators.find(materialType);
        if(loc == matGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_MATERIAL;

        GPUMatGPtr ptr = loc->second(gpuId);
        mg = ptr.get();
        matGroups.emplace(std::make_pair(materialType, gpuId), std::move(ptr));
    }
    else mg = loc->second.get();
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateMaterialBatch(GPUMaterialBatchI*& mb,
                                                       const GPUMaterialGroupI& mg,
                                                       const GPUPrimitiveGroupI& pg,
                                                       uint32_t keyBatchId)
{
    mb = nullptr;
    if(matBatchMap.find(keyBatchId) != matBatchMap.end())
        return SceneError::INTERNAL_DUPLICATE_MAT_ID;

    const std::string batchType = std::string(mg.Type()) + pg.Type();

    auto loc = matBatches.find(std::make_pair(batchType, mg.GPUId()));
    if(loc == matBatches.end())
    {
        // Cannot Find Already Constructed Type
        // Generate
        auto loc = matBatchGenerators.find(batchType);
        if(loc == matBatchGenerators.end()) return SceneError::NO_LOGIC_FOR_MATERIAL;

        GPUMatBPtr ptr = loc->second(mg, pg);
        mb = ptr.get();
        matBatches.emplace(std::make_pair(batchType, mg.GPUId()), std::move(ptr));
        matBatchMap.emplace(keyBatchId, mb);
    }
    else mb = loc->second.get();
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateBaseAccelerator(GPUBaseAcceleratorI*& baseAccel,
                                                         const std::string& accelType)
{
    if(baseAccelerator.get() == nullptr)
    {
        // Cannot Find Already Constructed Type
        // Generate
        auto loc = baseAccelGenerators.find(accelType);
        if(loc == baseAccelGenerators.end()) return SceneError::NO_LOGIC_FOR_ACCELERATOR;
        baseAccelerator = loc->second();
        baseAccel = baseAccelerator.get();
    }
    else baseAccel = baseAccelerator.get();
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateBaseLogic(TracerBaseLogicI*& bl,
                                                   // Args
                                                   const TracerParameters& opts,
                                                   const Vector2i maxMats,
                                                   const Vector2i maxAccels,
                                                   const HitKey baseBoundMatKey,
                                                   // DLL Location
                                                   const std::string& libName,
                                                   const SharedLibArgs& mangledName)
{

    // Do stuff (
    //.....

    uint32_t hitStructSize = CalculateHitStruct();

    auto ag = GetAcceleratorGroups();
    auto ab = GetAcceleratorBatches();
    auto mg = GetMaterialGroups();
    auto mb = GetMaterialBatches();

    //bl = nullptr;
    //if(tracerPtr == nullptr)
    //    tracerPtr = tracerGenerator(*baseAccelerator.get(),
    //                                  std::move(ag),
    //                                  std::move(ab),
    //                                  std::move(mg),
    //                                  std::move(mb),
    //                                  opts, hitStructSize,
    //                                  maxMats, maxAccels,
    //                                  baseBoundMatKey);
    bl = tracerPtr.get();

    return SceneError::OK;
}

PrimitiveGroupList TracerLogicGenerator::GetPrimitiveGroups() const
{
    std::vector<GPUPrimitiveGroupI*> result;
    for(const auto& p : primGroups)
    {
        result.push_back(p.second.get());
    }
    return std::move(result);
}

AcceleratorGroupList TracerLogicGenerator::GetAcceleratorGroups() const
{
    std::vector<GPUAcceleratorGroupI*> result;
    for(const auto& p : accelGroups)
    {
        result.push_back(p.second.get());
    }
    return std::move(result);
}

AcceleratorBatchMappings TracerLogicGenerator::GetAcceleratorBatches() const
{
    return std::move(accelBatchMap);
}

MaterialGroupList TracerLogicGenerator::GetMaterialGroups() const
{
    MaterialGroupList result;
    for(const auto& p : matGroups)
    {
        result.push_back(p.second.get());
    }
    return std::move(result);
}

MaterialBatchMappings TracerLogicGenerator::GetMaterialBatches() const
{
    return std::move(matBatchMap);
}

GPUBaseAcceleratorI* TracerLogicGenerator::GetBaseAccelerator() const
{
    return baseAccelerator.get();
}

void TracerLogicGenerator::ClearAll()
{
    primGroups.clear();

    accelGroups.clear();
    accelBatches.clear();

    matGroups.clear();
    matBatches.clear();

    baseAccelerator.reset(nullptr);
}

SceneError TracerLogicGenerator::IncludeBaseAcceleratorsFromDLL(const std::string& libName,
                                                                const SharedLibArgs& mangledName) const
{
    // TODO: Implement

    // Load Shared lib if not loaded (same dll may cover other stuff too)
    return SceneError::OK;
    // Then 
}
SceneError TracerLogicGenerator::IncludeAcceleratorsFromDLL(const std::string& libName,
                                                            const SharedLibArgs& mangledName) const
{
    // TODO: Implement
    return SceneError::OK;
}

SceneError TracerLogicGenerator::IncludeMaterialsFromDLL(const std::string& libName,
                                                         const SharedLibArgs& mangledName) const
{
    // TODO: Implement
    return SceneError::OK;
}

SceneError TracerLogicGenerator::IncludePrimitivesFromDLL(const std::string& libName,
                                                          const SharedLibArgs& mangledName) const
{
    // TODO: Implement
    return SceneError::OK;
}