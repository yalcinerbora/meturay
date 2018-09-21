#pragma once
/**

Tracer error "Enumeration"

*/

#include "Error.h"

struct TracerError : public ErrorI
{
	public:
		enum Type
		{
			OK,
			// General
			OUT_OF_MEMORY,
			// ...



			// End
			END
		};

	private:
		Type			 type;

	public:
		// Constructors & Destructor 
					TracerError(Type);
					~TracerError() = default;

		operator	std::string() const override;
};

inline TracerError::TracerError(TracerError::Type t)
	: type(t)
{}

inline TracerError::operator std::string() const
{
	const char* const ErrorStrings[] =
	{
		"OK.",
		// General
		"File not found.",
	};
	static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(TracerError::END),
				  "Enum and enum string list size mismatch.");

	return ErrorStrings[static_cast<int>(type)];
}