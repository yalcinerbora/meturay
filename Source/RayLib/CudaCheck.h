/**

Utility header for header only cuda vector and cpu vector implementations

*/

#ifdef METU_CUDA
	#include <cuda.h>
	#include <cuda_runtime.h>
#else
	#define __device__
	#define __host__
#endif
