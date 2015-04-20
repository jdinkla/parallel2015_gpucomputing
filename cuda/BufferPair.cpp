/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <cuda_runtime_api.h>
#include "BufferPair.h"
#include "CudaUtilities.h"
#include <assert.h>
#include "Extent2.h"
#include "Extent3.h"

template <typename T, class E>
BufferPair<T, E>::BufferPair(PinnedBuffer<T, E>& _host, DeviceBuffer<T, E>& _device)
	: host(_host)
	, device(_device)
{
}

template <typename T, class E>
void BufferPair<T, E>::update_host()
{
	if (host.get_version() < device.get_version())
	{
		device.copy_to(host);
	}
}

template <typename T, class E>
void BufferPair<T, E>::update_device()
{
	if (host.get_version() > device.get_version())
	{
		device.copy_from(host);
	}
}

// Instances
template class BufferPair < int, Extent2 >;
template class BufferPair < float, Extent2 >;
template class BufferPair < double, Extent2 >;
template class BufferPair < uchar4, Extent2 >;

template class BufferPair < float, Extent3 >;
template class BufferPair < int, Extent3 >;
