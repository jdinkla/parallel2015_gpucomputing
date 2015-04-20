/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include "Extent2.h"
#include <iostream>

class Extent3
{

public:

	__device__ __host__
	Extent3(const int _width = 1, const int _height = 1, const int _depth = 1)
		: width(_width)
		, height(_height)
		, depth(_depth)
	{
	}

	__device__ __host__
	Extent3(const Extent3& extent)
		: width(extent.width)
		, height(extent.height)
		, depth(extent.depth)
	{
	}

	__device__ __host__
	Extent3(const Extent2& extent)
		: width(extent.get_width())
		, height(extent.get_height())
		, depth(1)
	{
	}

	__device__ __host__
	int index(const int x, const int y, const int z) const
	{
		return z * (width * height) + y * width + x;
	}

	__device__ __host__
	int checked_index(const int x, const int y = 0, const int z = 0) const
	{
		int result = -1;
		if (0 <= x && x < width && 0 <= y && y < height && 0 <= z && z < depth)
		{
			result = z * (width * height) + y * width + x;
		}
		return result;
	}

	__device__ __host__
	bool in_bounds(const int x, const int y = 0, const int z = 0) const
	{
		return x < width && y < height && z < depth;
	}

	__device__ __host__
	bool in_bounds_strict(const int x, const int y = 0, const int z = 0) const
	{
		return 0 <= x && x < width && 0 <= y && y < height && 0 <= z && z < depth;
	}

	__device__ __host__
	int get_width() const
	{
		return width;
	}

	__device__ __host__
	int get_height() const
	{
		return height;
	}

	__device__ __host__
	int get_depth() const
	{
		return width;
	}

	__device__ __host__
	int get_number_of_elems() const
	{
		return width * height * depth;
	}

	__device__ __host__
	Extent3& operator=(const Extent3& a)
	{
		width = a.width;
		height = a.height;
		depth = a.depth;
		return *this;
	}

	__device__ __host__
	bool operator==(const Extent3& b) const
	{
		return this->width == b.width && this->height == b.height && this->depth == b.depth;
	}

private:

	int width;
	int height;
	int depth;

};

inline std::ostream &operator<<(std::ostream& ostr, const Extent3& d)
{
	return ostr << d.get_width() << "," << d.get_height() << "," << d.get_depth();
}
