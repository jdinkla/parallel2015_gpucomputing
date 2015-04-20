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
#include "Stencil.h"
#include <memory>

#ifdef __CUDACC__

template <typename T>
__global__
void stencil_heat_2d_kernel(const T* src, T* dest, Extent2 ext, const T value)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (ext.in_bounds(x, y))
	{
		auto get = [&ext, &src](const int x, const int y, const T default_value)
		{
			return ext.in_bounds_strict(x, y) ? src[ext.index(x, y)] : default_value;
		};
		const int idx = ext.index(x, y);
		const T c = src[idx];
		const T l = get(x - 1, y, c);
		const T r = get(x + 1, y, c);
		const T t = get(x, y - 1, c);
		const T b = get(x, y + 1, c);
		dest[idx] = c + value * (t + b + l + r - 4 * c);
	}
}

#endif

template <typename T>
void stencil_heat_2d(
	const CudaExecConfig& cnf,
	DeviceBuffer<T>& srcBuf,
	DeviceBuffer<T>& destBuf,
	const T value
	);

template <typename T>
void stencil_heat_2d(
	const CudaExecConfig& cnf,
	std::shared_ptr<DeviceBuffer<T>> src,
	std::shared_ptr<DeviceBuffer<T>> dst, const T value);

template <typename T>
void stencil_heat_2d(
	const CudaExecConfig& cnf,
	thrust::device_vector<T>& srcVec,
	thrust::device_vector<T>& destVec,
	const T value
	);



