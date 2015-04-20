/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "HostBuffer.h"
#include "AsyncScheduler.h"

template <typename T> inline
void stencil_heat_2d_row(const T* src, T* dest, const Extent2 &ext, const T value, const int y)
{
	const int w = ext.get_width();
	int idx = ext.index(0, y);
	auto get = [&ext, &src](const int x, const int y, const T default_value)
	{
		return ext.in_bounds_strict(x, y) ? src[ext.index(x, y)] : default_value;
	};
	for (int x = 0; x < w; x++)
	{
		const T c = src[idx];
		const T l = get(x - 1, y, c);
		const T r = get(x + 1, y, c);
		const T t = get(x, y - 1, c);
		const T b = get(x, y + 1, c);
		dest[idx] = c + value * (t + b + l + r - 4 * c);
		idx++;
	}
}

template <typename T> inline
void stencil_heat_2d(HostBuffer<T>& src, HostBuffer<T>& dest, const T value)
{
	const Extent2& ext = src.get_extent();
	const int h = ext.get_height();
	T* src_ptr = src.get_ptr();
	T* dest_ptr = dest.get_ptr();

	for (int y = 0; y < h; y++)
	{
		stencil_heat_2d_row(src_ptr, dest_ptr, ext, value, y);
	}
}

template <typename T> inline
void stencil_heat_2d_par(HostBuffer<T>& src, HostBuffer<T>& dest, const T value, 
	const ISchedulerInterval<int>& s = AsyncSchedulerInterval<int>())
{
	const Extent2& ext = src.get_extent();
	const int h = ext.get_height();
	T* src_ptr = src.get_ptr();
	T* dest_ptr = dest.get_ptr();

	s.sync(0, h, [src_ptr, dest_ptr, ext, value](const int y)
	{
		stencil_heat_2d_row(src_ptr, dest_ptr, ext, value, y);
	});
}

