/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Extent2.h"
#include "CudaExecConfig.h"
#include "DeviceBuffer.h"

template <typename T> inline __device__ __host__ 
void mask(const T* mask, T* dest, Extent2 &ext, const int x, const int y)
{
	const int idx = ext.index(x, y);
	const float value = mask[idx];
	if (value)
	{
		dest[idx] = value;
	}
}

template <typename T>
__global__
void mask_2d_kernel(const T* mask, T* dest, Extent2 ext)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (ext.in_bounds(x, y))
	{
		const int idx = ext.index(x, y);
		const float value = mask[idx];
		if (value)
		{
			dest[idx] = value;
		}
	}
}

template <typename T>
void mask_2d(
	const CudaExecConfig& cnf,
	DeviceBuffer<T>& mask,
	DeviceBuffer<T>& dest
	)
{
	mask_2d_kernel << <cnf.get_grid(), cnf.get_block(), 0, cnf.get_stream() >> >(
		mask.get_ptr(),
		dest.get_ptr(),
		dest.get_extent());
}

template <typename T>
void mask_2d(
	const CudaExecConfig& cnf,
	std::shared_ptr<DeviceBuffer<T>> mask,
	std::shared_ptr<DeviceBuffer<T>> dest
	)
{
	mask_2d(cnf, *mask.get(), *dest.get());
}


