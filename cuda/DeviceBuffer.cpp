/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <cuda_runtime_api.h>
#include "Defs.h"
#include "DeviceBuffer.h"
#include "CudaUtilities.h"
#include <cassert>
#include "Extent2.h"
#include "Extent3.h"

template <typename T, class E>
DeviceBuffer<T, E>::DeviceBuffer(const E& e)
	: BaseBuffer<T, E>(e)
{
}

template <typename T, class E>
DeviceBuffer<T, E>::~DeviceBuffer()
{
	if (ptr)
	{
		free();
	}
}

template <typename T, class E>
void DeviceBuffer<T, E>::alloc()
{
	cudaMalloc((void**)&ptr, BaseBuffer<T, E>::get_size_in_bytes());
	CUDA::check("cudaMalloc");
	version = 0;
}

template <typename T, class E>
void DeviceBuffer<T, E>::free()
{
	cudaFree(ptr);
	CUDA::check("cudaFree");
	version = -1;
	ptr = 0;
}

template <typename T, class E>
void DeviceBuffer<T, E>::copy_from(const PinnedBuffer<T, E>& buf)
{
	const E ext1 = buf.get_extent();
	const E ext2 = BaseBuffer<T, E>::get_extent();
	assert(ext1 == ext2);
	const size_t sz = BaseBuffer<T, E>::get_size_in_bytes();
	//void* dst = BaseBuffer<T, E>::get_ptr();
	const void* src = buf.get_ptr();
	cudaMemcpy(ptr, src, sz, cudaMemcpyHostToDevice);
	CUDA::check("cudaMemcpy");
	BaseBuffer<T, E>::set_version(buf);
}

template <typename T, class E>
void DeviceBuffer<T, E>::copy_to(PinnedBuffer<T, E>& buf) const
{
	const E ext1 = buf.get_extent();
	const E ext2 = BaseBuffer<T, E>::get_extent();
	assert(ext1 == ext2);
	const size_t sz = BaseBuffer<T, E>::get_size_in_bytes();
	void* dst = buf.get_ptr();
	//const void* src = BaseBuffer<T, E>::get_ptr();
	cudaMemcpy(dst, ptr, sz, cudaMemcpyDeviceToHost);
	CUDA::check("cudaMemcpy");
	buf.set_version(*this);
}

// Instances
template class DeviceBuffer<int, Extent2>;
template class DeviceBuffer<float, Extent2>;
template class DeviceBuffer<double, Extent2>;
template class DeviceBuffer<uchar4, Extent2>;

template class DeviceBuffer<float, Extent3>;
template class DeviceBuffer<int, Extent3>;


