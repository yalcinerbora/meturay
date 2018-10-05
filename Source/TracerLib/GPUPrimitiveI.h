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

#include <vector>
#include <string>

struct SceneError;

struct EditList
{
	uint32_t index;
	nlohmann::json ...;
};

class GPUPrimitiveGroupI
{
	public:
		virtual						~GPUPrimitiveGroupI() = default;

		// Interface
		// Pirmitive type is used for delegating scene info to this class
		virtual const std::string&		PrimitiveType() const = 0;

		// Allocates and Generates Data
		virtual SceneError				InitializeData(const std::vector<...>& surfaceDataList,
													   double time) = 0;
		// Load
		virtual SceneError				RefreshData(const std::vector<...>& editedDataList, 
													double time) = 0;
	
		// Error check
		// Queries in order to check if this primitive group supports certain primitive data
		// Material may need this
		virtual bool					CanGenerateData(const std::string& s) const = 0;
};