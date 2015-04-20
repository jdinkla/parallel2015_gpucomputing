/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <cuda_runtime_api.h>
#include "ManagedBuffer.h"
#include "CudaUtilities.h"
#include <assert.h>
#include "Defs.h"
#include "Extent2.h"
#include "Extent3.h"

template <typename T, class E>
ManagedBuffer<T, E>::ManagedBuffer(const E& e)
	: HostBuffer<T, E>(e)
{
}

template <typename T, class E>
ManagedBuffer<T, E>::~ManagedBuffer()
{
	if (ptr)
	{
		free();
	}
}

template <typename T, class E>
void ManagedBuffer<T, E>::alloc()
{
	const size_t sz = BaseBuffer<T, E>::get_size_in_bytes();
	cudaMallocManaged((void**)&ptr, sz, cudaMemAttachHost);
	version = 0;
	CUDA::check("cudaMallocManaged");
}

template <typename T, class E>
void ManagedBuffer<T, E>::free()
{
	if (ptr)
	{
		cudaFree(ptr);
		version = -1;
		ptr = nullptr;
		CUDA::check("cudaFree");
	}
}

// Instances
template class ManagedBuffer < int, Extent2 >;
template class ManagedBuffer < float, Extent2 >;
template class ManagedBuffer < double, Extent2 >;
template class ManagedBuffer < uchar4, Extent2 >;

template class ManagedBuffer < float, Extent3 >;
template class ManagedBuffer < int, Extent3 >;
