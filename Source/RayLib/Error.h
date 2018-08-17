#pragma once
/**

Generic Error

*/

#include <cstdint>
#include "IOError.h"
#include "TracerError.h"

enum class ErrorType
{
	ANY_ERROR,
	IO_ERROR,
	TRACER_ERROR,
	DISTRIBUTOR_ERROR,
};

struct Error
{
	static constexpr	uint32_t OK = 0;

	ErrorType			t;
	uint32_t			errorCode;

	public:
		operator		IOError() const;
};

static_assert(static_cast<uint32_t>(IOError::OK) == Error::OK, 
			  "All error types should have same value of OK.");


inline Error::operator IOError() const
{
	assert(t == ErrorType::IO_ERROR);
	return static_cast<IOError>(errorCode);
}

typedef void(*ErrorCallbackFunction)(Error);
