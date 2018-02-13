/**

Utility header for header only cuda vector and cpu vector implementations

*/

#ifdef METU_CUDA
	#include <cuda.h>
	#include <cuda_runtime.h>
	#define METU_UNROLL #pragma unroll
#else
	#define __device__
	#define __host__
	#define METU_UNROLL
#endif

#ifdef __CUDA_ARCH__
	#define UNROLL_LOOP #pragma unroll
#else 
	#define UNROLL_LOOP
#endif

#ifdef METU_DEBUG
	#define CUDA_CHECK
#else
#endif