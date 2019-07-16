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
class SceneNodeI;
class SurfaceLoaderGeneratorI;

class GPUPrimitiveGroupI
{
    public: 
        virtual                     ~GPUPrimitiveGroupI() = default;

        // Interface
        // Type (as string) of the primitive group
        virtual const char*         Type() const = 0;
        // Allocates and Generates Data
        virtual SceneError          InitializeGroup(const NodeListing& surfaceDatalNodes, double time,
                                                    const SurfaceLoaderGeneratorI&) = 0;
        virtual SceneError          ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                               const SurfaceLoaderGeneratorI&) = 0;

        // Access primitive range from Id     
        virtual Vector2ul           PrimitiveBatchRange(uint32_t surfaceDataId) const = 0;
        virtual AABB3               PrimitiveBatchAABB(uint32_t surfaceDataId) const = 0;

        virtual uint32_t            PrimitiveHitSize() const = 0;

        // Error check
        // Queries in order to check if this primitive group supports certain primitive data
        // Material may need that data
        virtual bool                CanGenerateData(const std::string& s) const = 0;
};