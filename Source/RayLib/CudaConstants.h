#pragma once

/**

Compile Time Cuda Constants

Thread per Block etc..

*/

#include "Vector.h"

// Thread Per Block
static constexpr int TPB_1D = 512;					
static constexpr Vector2i TPB_2D = Vector2i(16, 16);