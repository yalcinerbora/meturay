#pragma once

#include <array>

constexpr int StrLenCT(const char* str)
{
	return (*str) ? (1 + StrLenCT(str + 1)) : 0;
}

template<int Size0, int Size1>
constexpr auto StrConcatCT(const char* str0, const char* str1)
{
	constexpr int Size = Size0 + Size1 + 1;

	std::array<char, Size> result = {};
	for(int i = 0; i < Size0; i++)
	{
		result[i] = str0[i];
	}
	for(int i = 0; i < Size1; i++)
	{
		result[i + Size0] = str0[i];
	}
	result[Size - 1] = '\0';
	return result;
}
