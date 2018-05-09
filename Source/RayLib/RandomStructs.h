#pragma once

#include <cstdint>

struct RandomStackGMem
{
	uint32_t* state;

	constexpr operator ConstRandomStackGMem() const;
};

struct ConstRandomStackGMem
{
	const uint32_t* state;

	
};

constexpr RandomStackGMem::operator ConstRandomStackGMem() const
{
	return {state};
}
