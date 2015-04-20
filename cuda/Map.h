/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Extent2.h"
#include "DeviceBuffer.h"
#include "CudaExecConfig.h"
#include <thrust/device_vector.h>
#include "ManagedBuffer.h"

template <typename I, typename O, class F>
__global__
void map_2d_kernel(const I* src, O* dest, Extent2 ext, const F functor)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (ext.in_bounds(x, y))
	{
		const int idx = ext.index(x, y);
		dest[idx] = functor(src[idx]);
	}
}

template <typename I, typename O, class F>
void map_2d(
	const CudaExecConfig& cnf,
	const I* src, 
	O* dst,
	Extent2& e,
	const F& functor)
{
	map_2d_kernel << <cnf.get_grid(), cnf.get_block(), 0, cnf.get_stream() >> >(src, dst, e, functor);
}

template <typename I, typename O, class F>
void map_2d(
	const CudaExecConfig& cnf,
	DeviceBuffer<I, Extent2>& srcBuf,
	DeviceBuffer<O, Extent2>& destBuf,
	const F& functor
	)
{
	const I* src = srcBuf.get_ptr();
	O* dst = destBuf.get_ptr();
	Extent2 e = srcBuf.get_extent();
	map_2d_kernel << <cnf.get_grid(), cnf.get_block(), 0, cnf.get_stream() >> >(src, dst, e, functor);
}

template <typename I, typename O, class F>
void map_2d(
	const CudaExecConfig& cnf,
	ManagedBuffer<I, Extent2>& srcBuf,
	ManagedBuffer<O, Extent2>& destBuf,
	const F& functor
	)
{
	const I* src = srcBuf.get_ptr();
	O* dst = destBuf.get_ptr();
	Extent2 e = srcBuf.get_extent();
	map_2d_kernel << <cnf.get_grid(), cnf.get_block(), 0, cnf.get_stream() >> >(src, dst, e, functor);
}

template <typename I, typename O, class F>
void map_2d(
	const CudaExecConfig& cnf,
	thrust::device_vector<I>& srcVec,
	thrust::device_vector<O>& destVec,
	const F& functor
	)
{
#ifdef SIMPLE
	const float* src = thrust::raw_pointer_cast(&srcVec[0]);
	float* dst = thrust::raw_pointer_cast(&destVec[0]);
	map_2d_kernel << <cnf.get_grid(), cnf.get_block() >> >(src, dst, cnf.get_extent(), functor);
#else
	const float* src = thrust::raw_pointer_cast(&srcVec[0]);
	float* dst = thrust::raw_pointer_cast(&destVec[0]);
	map_2d_kernel << <cnf.get_grid(), cnf.get_block(), 0, cnf.get_stream() >> >(src, dst, cnf.get_extent(), functor);
#endif
}

/*

#include <crt/nvfunctional>

void map_2d(
	DeviceBuffer<float, Extent2>& srcBuf,
	DeviceBuffer<float, Extent2>& destBuf,
	nvstd::function<float(const float)> f
	);

*/

