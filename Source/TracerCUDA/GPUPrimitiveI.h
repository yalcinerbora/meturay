#pragma once
/**

Primitive related structs

Primitive is a simple building block of a surface.
It can be numeric types (such as triangles, volumes etc.)
or it can be analtyic types (such as splines, spheres)

PrimtiveGroup holds multiple primitive lists (i.e. multiple meshes)

PrimtiveGroup holds the same primitives that have the same layout in memory
multiple triangle layouts will be on different primtive groups (this is required since
their primtiive data fetch logics will be different)

Most of the time user will define a single primtive for same types to have better performance
since this API is being developed for customization this is mandatory.

*/

#include <cstdint>

#include "RayLib/Vector.h"
#include "RayLib/AABB.h"

#include "NodeListing.h"

struct SceneError;
struct CPULight;

class SceneNodeI;
class SurfaceLoaderGeneratorI;
class CudaSystem;

enum class PrimTransformType : uint8_t
{
    // Single transform is applied to the group of primitives
    CONSTANT_LOCAL_TRANSFORM,
    // Each primitive will different (maybe) multiple transforms
    // this prevents the accelerator to utilize local space
    // accelerator structure and transforming ray to local space
    // instead of opposite
    PER_PRIMITIVE_TRANSFORM
};

class GPUPrimitiveGroupI
{
    public:
        virtual                     ~GPUPrimitiveGroupI() = default;

        // Interface
        // Type (as string) of the primitive group
        virtual const char*         Type() const = 0;
        // Allocates and Generates Data
        virtual SceneError          InitializeGroup(const NodeListing& surfaceDataNodes, double time,
                                                    const SurfaceLoaderGeneratorI&,
                                                    const TextureNodeMap& textureNodes,
                                                    const std::string& scenePath) = 0;
        virtual SceneError          ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                               const SurfaceLoaderGeneratorI&,
                                               const std::string& scenePath) = 0;
        // Access primitive range from Id
        virtual Vector2ul           PrimitiveBatchRange(uint32_t surfaceDataId) const = 0;
        virtual AABB3               PrimitiveBatchAABB(uint32_t surfaceDataId) const = 0;
        virtual uint32_t            PrimitiveHitSize() const = 0;
        virtual bool                PrimitiveBatchHasAlphaMap(uint32_t surfaceDataId) const = 0;
        virtual bool                PrimitiveBatchBackFaceCulled(uint32_t surfaceDataId) const = 0;

        // Primitive Transform Info for accelerator
        virtual PrimTransformType   TransformType() const = 0;
        // If primitive (by definition) is intersectable or not
        virtual bool                IsIntersectable() const = 0;
        // If this primitive group consists of tiangle
        // It may be usefull when a code (i.e. OptiX) can be optimized
        // for triangles
        virtual bool                IsTriangle() const = 0;

        // Query
        // How many primitives are available on this class
        // This includes the indexed primitive count
        virtual uint64_t            TotalPrimitiveCount() const = 0;
        // Total primitive count but not indexed
        virtual uint64_t            TotalDataCount() const = 0;

        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        virtual bool                CanGenerateData(const std::string& s) const = 0;
};