/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Extent2.h"
#include "HostBuffer.h"
#include "AsyncSchedulerInterval.h"

template <typename T> inline
void mask_2d_row(const T* mask, T* dest, const Extent2 &ext, const int y)
{
	const int w = ext.get_width();
	int idx = ext.index(0, y);
	for (int x = 0; x < w; x++)
	{
		const float value = mask[idx];
		if (value)
		{
			dest[idx] = value;
		}
		idx++;
	}
}

template <typename T> inline
void mask_2d(HostBuffer<T>& src, HostBuffer<T>& dest)
{
	const Extent2& ext = src.get_extent();
	const int h = ext.get_height();
	T* src_ptr = src.get_ptr();
	T* dest_ptr = dest.get_ptr();

	for (int y = 0; y < h; y++)
	{
		mask_2d_row(src_ptr, dest_ptr, ext, y);
	}
}

template <typename T> inline
void mask_2d_par(HostBuffer<T>& src, HostBuffer<T>& dest, 
	const ISchedulerInterval<int>& s = AsyncSchedulerInterval<int>())
{
	const Extent2& ext = src.get_extent();
	const int h = ext.get_height();
	T* src_ptr = src.get_ptr();
	T* dest_ptr = dest.get_ptr();

	s.sync(0, h, [src_ptr, dest_ptr, ext](const int y)
	{
		mask_2d_row(src_ptr, dest_ptr, ext, y);
	});
	
}
