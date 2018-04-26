#pragma once

#include <vector>
#include <string>
#include "Vector.h"

namespace ImageIO
{
	bool WriteAsPNG(const std::vector<Vector3>& image,
					const Vector2ui& size,
					const std::string& fileName);
}