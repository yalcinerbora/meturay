#pragma once
/**

Basic output image memory
Used by tracers

*/

#include <vector>

#include "RayLib/Types.h"
#include "RayLib/Vector.h"

#include "DeviceMemory.h"

class ImageMemory
{
	private:
		DeviceMemory			memory;
		PixelFormat				format;
		size_t					pixelSize;

		Vector2i				segmentSize;
		Vector2i				segmentOffset;
		Vector2i				resolution;

		static size_t			PixelFormatToSize(PixelFormat);

	protected:
	public:
		// Constructors & Destructors
								ImageMemory();
								ImageMemory(const Vector2i& offset,
											const Vector2i& size,
											const Vector2i& resolution,
											PixelFormat f);
								ImageMemory(const ImageMemory&) = delete;
								ImageMemory(ImageMemory&&) = default;
		ImageMemory&			operator=(const ImageMemory&) = delete;
		ImageMemory&			operator=(ImageMemory&&) = default;
								~ImageMemory() = default;
					
		void					SetPixelFormat(PixelFormat);
		void					Reportion(Vector2i offset,
										  Vector2i size);
		void					Resize(Vector2i resolution);
		void					Reset();

		template<class T>
		std::vector<T>			MoveImageToCPU();
	
		// Getters
		Vector2i				SegmentSize() const;
		Vector2i				SegmentOffset() const;
		Vector2i				Resolution() const;
		
		template <class T>
		T*						GMem();
};

template<class T>
inline std::vector<T> ImageMemory::MoveImageToCPU()
{
	size_t pixelCount = segmentSize[0] * segmentSize[1];
	std::vector<T> out(pixelCount);
	std::memcpy(out.data(), imagePtr, sizeof(T) * pixelCount);
	return std::move(out);
}

template<class T>
inline T* ImageMemory::GMem()
{
	return static_cast<T*>(memory);
}

