/*
 * Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
 * 
 * See the LICENSE file in the root directory.
 */

#pragma once

#include "Defs.h"
#include "Extent2.h"

#ifdef __CUDACC__
template <typename T> inline __device__ __host__
T get_with_default(const T* src, Extent2& ext, const int x, const int y, const T default_value)
{
	return ext.in_bounds_strict(x, y) ? src[ext.index(x, y)] : default_value;
}

template <typename I, typename O, class F>
__global__
void stencil_2d_kernel(const I* src, O* dest, Extent2 ext, F functor)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (ext.in_bounds(x, y))
	{
		dest[ext.index(x, y)] = functor(src, dest, x, y);
	}
}
#endif

/*
#include "Extent2.h"
#include "DeviceBuffer.h"
#include "CudaExecConfig.h"

#include <thrust/device_vector.h>


template <typename I, typename O>
__global__
void stencil_average_2d_kernel(const I* src, O* dest, Extent2 ext)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = ext.checked_index(x, y);
	if (ext.in_bounds(x, y))
	{
		O sum = O(0);
		int count = 0;
		for (int dy = -1; dy <= 1; dy++)
		{
			for (int dx = -1; dx <= 1; dx++)
			{
				if (ext.in_bounds_strict(x + dx, y + dy))
				{
					sum += src[ext.index(x + dx, y + dy)];
					count++;
				}
			}
		}
		dest[ext.index(x, y)] = sum / count;
	}
}

template <typename I, typename O, class F>
void stencil_2d(
	CudaExecConfig& cnf,
	DeviceBuffer<float, Extent2>& srcBuf,
	DeviceBuffer<float, Extent2>& destBuf,
	const F& op
	)
{
	const float* src = srcBuf.get_ptr();
	float* dst = destBuf.get_ptr();
	Extent2 e = srcBuf.get_extent();
	stencil_average_2d_kernel << <cnf.get_grid(), cnf.get_block() >> >(op, e, src, dst);
}

template <typename I, typename O, class F>
void stencil_2d(
	CudaExecConfig& cnf,
	thrust::device_vector<I>& srcVec,
	thrust::device_vector<O>& destVec,
	const F& op
	)
{
	const float* src = thrust::raw_pointer_cast(&srcVec[0]);
	float* dst = thrust::raw_pointer_cast(&destVec[0]);
	stencil_average_2d_kernel << <cnf.get_grid(), cnf.get_block() >> >(src, dst, cnf.get_extent(), op);
}



*/
