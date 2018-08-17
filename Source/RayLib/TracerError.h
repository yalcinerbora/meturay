#pragma once

enum class TracerError
{
	OK,
	// General
	OUT_OF_MEMORY,
	// ...



	// End
	END
};

static constexpr const char* GetTracerErrorErrorString(TracerError e)
{
	constexpr const char* ErrorStrings[] =
	{
		"OK.",
		// General
		"File not found.",
	};
	static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(TracerError::END),
				  "Enum and enum string list size mismatch.");

	return ErrorStrings[static_cast<int>(e)];
}