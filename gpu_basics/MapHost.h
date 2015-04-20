/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Extent2.h"
#include "HostBuffer.h"
#include <functional>
#include "AsyncSchedulerInterval.h"

template <typename I, typename O> inline
void map_2d_row(const I* src_ptr, O* dest_ptr, const Extent2 &ext, std::function<O(I)> f, const int y)
{
	const int w = ext.get_width();
	int idx = ext.index(0, y);
	for (int x = 0; x < w; x++)
	{
		dest_ptr[idx] = f(src_ptr[idx]);
		idx++;
	}
}

template <typename I, typename O> inline
void map_2d(HostBuffer<I>& src, HostBuffer<O>& dest, std::function<O(I)> f)
{
	Extent2& ext = src.get_extent();
	const int h = ext.get_height();
	I* src_ptr = src.get_ptr();
	O* dest_ptr = dest.get_ptr();

	for (int y = 0; y < h; y++)
	{
		map_2d_row(src_ptr, dest_ptr, ext, f, y);
	}
}

template <typename I, typename O> inline
void map_2d(const I* src, O* dest, Extent2& ext, std::function<O(I)> f)
{
	const int h = ext.get_height();
	for (int y = 0; y < h; y++)
	{
		map_2d_row(src, dest, ext, f, y);
	}
}

template <typename I, typename O> inline
void map_2d_par(
	I* src, 
	O* dest, 
	Extent2& ext, 
	std::function<O(I)> f, 
	const ISchedulerInterval<int>& s = AsyncSchedulerInterval<int>())
{
	const int h = ext.get_height();
	s.sync(0, h, [src, dest, ext, f](const int y)
	{
		map_2d_row(src, dest, ext, f, y);
	});
}



