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
void mask(const T* mask, T* dest, Extent2 &ext_src, Extent2 &ext_dest, Pos2 &offset, const int x, const int y)
{
	const int idx_src = ext_src.index(x, y);
	const float value = mask[idx_src];
	if (value)
	{
		const int idx_dest = ext_dest.index(x + offset.x, y + offset.y);
		dest[idx_dest] = value;
	}
}

template <typename T>
__global__
void mask_2d_kernel(const T* mask, T* dest, Extent2 ext_src, 
	Extent2 ext_dest, Pos2 offset)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (ext_src.in_bounds(x, y))
	{
		const int idx_src = ext_src.index(x, y);
		const float value = mask[idx_src];
		if (value)
		{
			const int idx_dest = ext_dest.index(x + offset.x, y + offset.y);
			dest[idx_dest] = value;
		}
	}
}

template <typename T>
void mask_2d(
	const CudaExecConfig& cnf,
	DeviceBuffer<T>& mask,
	DeviceBuffer<T>& dest,
	const Pos2& offset
	)
{
	mask_2d_kernel << <cnf.get_grid(), cnf.get_block(), 0, cnf.get_stream() >> >(
		mask.get_ptr(),
		dest.get_ptr(),
		mask.get_extent(),
		dest.get_extent(),
		offset
		);
}

template <typename T>
void mask_2d(
	const CudaExecConfig& cnf,
	std::shared_ptr<DeviceBuffer<T>> mask,
	std::shared_ptr<DeviceBuffer<T>> dest,
	const Pos2& offset
	)
{
	mask_2d(cnf, *mask.get(), *dest.get(), offset);
}


