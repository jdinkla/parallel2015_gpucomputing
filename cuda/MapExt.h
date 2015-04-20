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

template <typename O, class F>
__global__
void map_ext_2d_kernel(O* dest, Extent2 ext, const F functor)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = ext.checked_index(x, y);
	if (i >= 0)
	{
		dest[i] = functor(x, y);
	}
}

template <typename O, class F>
void map_ext_2d(
	const CudaExecConfig& cnf,
	thrust::device_vector<O>& destVec,
	const F& op
	)
{
	float* dst = thrust::raw_pointer_cast(&destVec[0]);
	map_ext_2d_kernel << <cnf.get_grid(), cnf.get_block() >> >(dst, cnf.get_extent(), op);
}


