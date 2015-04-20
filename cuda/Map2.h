/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Extent2.h"
#include "CudaExecConfig.h"
#include "DeviceBuffer.h"

template <typename T, class F>
__global__
void map_2d_kernel(const T* src_a, const T* src_b, T* dest, Extent2 ext, const F functor)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (ext.in_bounds(x, y))
	{
		const int idx = ext.index(x, y);
		const float value_a = src_a[idx];
		const float value_b = src_b[idx];
		dest[idx] = functor(value_a, value_b);
	}
}

template <typename T, class F>
void map_2d(
	const CudaExecConfig& cnf,
	DeviceBuffer<T>& src_a,
	DeviceBuffer<T>& src_b,
	DeviceBuffer<T>& dest,
	const F& functor
	)
{
	map_2d_kernel << <cnf.get_grid(), cnf.get_block(), 0, cnf.get_stream() >> >(
		src_a.get_ptr(),
		src_b.get_ptr(),
		dest.get_ptr(),
		dest.get_extent(),
		functor);
}

