/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "PinnedBuffer.h"
#include "CudaUtilities.h"
#include "Extent2.h"
#include "Extent3.h"

template <typename T, class E>
PinnedBuffer<T, E>::PinnedBuffer(const E& e)
	: HostBuffer<T, E>(e)
{
}

template <typename T, class E>
PinnedBuffer<T, E>::~PinnedBuffer()
{
	if (ptr)
	{
		free();
	}
}

template <typename T, class E>
void PinnedBuffer<T, E>::alloc()
{
	const size_t sz = BaseBuffer<T, E>::get_size_in_bytes();
	cudaMallocHost((void**)&ptr, sz);
	CUDA::check("cudaMallocHost");
	version = 0;
}
 
template <typename T, class E>
void PinnedBuffer<T, E>::free()
{
	if (ptr)
	{
		cudaFreeHost(ptr);
		ptr = 0;
		CUDA::check("cudaFreeHost");
		version = -1;
	}
}

// Instances
template class PinnedBuffer < int, Extent2 >;
template class PinnedBuffer < float, Extent2 >;
template class PinnedBuffer < double, Extent2 >;
template class PinnedBuffer < uchar4, Extent2 >;

template class PinnedBuffer < float, Extent3 >;
template class PinnedBuffer < int, Extent3 >;

#if defined(LINUX)
template class HostBuffer < int, Extent2 >;
template class HostBuffer < float, Extent2 >;
template class HostBuffer < double, Extent2 >;
template class HostBuffer < uchar4, Extent2 >;

template class HostBuffer < float, Extent3 >;
template class HostBuffer < int, Extent3 >;
#endif
