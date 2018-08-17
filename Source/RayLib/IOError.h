#pragma once
/**

I-O Error Enumeration


*/

enum class IOError
{	
	OK,
	// General
	FILE_NOT_FOUND,
	// Scene
	SCENE_CORRUPTED,
	// Maya nCache
	NCACHE_XML_ERROR,
	NCACHE_INVALID_FOURCC,
	NCACHE_INVALID_FORMAT,
	// Maya nCache Navier-Stokes Fluid
	NCACHE_DENSITY_NOT_FOUND,
	NCACHE_VELOCITY_NOT_FOUND,
	// Scene Json
	//....

	// End
	END
};

static constexpr const char* GetIOErrorString(IOError e)
{
	constexpr const char* ErrorStrings[] = 
	{		
		"OK.",
		// General
		"File not found.",
		// Scene
		"Scene file is corrupted.",
		// Maya nCache
		"nCache XML parse error.",
		"nCache invalid fourcc code.",
		"nCache invalid file format code.",
		// Maya nCache Navier-Stokes Fluid
		"nCache \"density\" channel not found.",
		"nCache \"velocity\" channel not found."
	};
	static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(IOError::END), 
				  "Enum and enum string list size mismatch.");

	return ErrorStrings[static_cast<int>(e)];
}