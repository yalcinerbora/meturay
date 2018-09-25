#pragma once
/**

Basic output image memory

*/

#include <vector>

#include "RayLib/DeviceMemory.h"
#include "RayLib/Types.h"
#include "Vector.h"

class ImageMemory
{
	private:
		DeviceMemory			memory;
		PixelFormat				format;
		size_t					pixelSize;

		Vector2ui				segmentSize;
		Vector2ui				segmentOffset;
		Vector2ui				resolution;

		static size_t			PixelFormatToSize(PixelFormat);

	protected:
	public:
		// Constructors & Destructors
								ImageMemory() = default;
								ImageMemory(const Vector2ui& offset,
											const Vector2ui& size,
											const Vector2ui& resolution,
											PixelFormat f);
								ImageMemory(const ImageMemory&) = delete;
								ImageMemory(ImageMemory&&) = default;
		ImageMemory&			operator=(const ImageMemory&) = delete;
		ImageMemory&			operator=(ImageMemory&&) = default;
								~ImageMemory() = default;
					
		void					SetPixelFormat(PixelFormat);
		void					Reportion(const Vector2ui& offset,
										  const Vector2ui& size);
		void					Resize(const Vector2ui& resolution);
		void					Reset();

		template<class T>
		std::vector<T>			MoveImageToCPU();
	
		// Getters
		Vector2ui				SegmentSize() const;
		Vector2ui				SegmentOffset() const;
		Vector2ui				Resolution() const;
		
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

