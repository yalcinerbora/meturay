#pragma once

#include <vector>
#include <string>
#include <memory>

#include "Vector.h"

class ImageIO
{
	private:
		static std::unique_ptr<ImageIO>	instance;

	protected:
	public:
		// Constructors & Destructor
										ImageIO();
										ImageIO(const ImageIO&) = delete;
		ImageIO&						operator=(const ImageIO&) = delete;
										~ImageIO();

		// Singleton Accessor
		static ImageIO&					System();
										
		// Usage
		// Read

		// Write
		bool							WriteAsPNG(const std::vector<Vector3>& image,
												   const Vector2ui& size,
												   const std::string& fileName) const;
};