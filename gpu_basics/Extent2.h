/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include <iostream>
#include <vector>
#include "PartitionUtilities.h"
#include "Pos2.h"

class Extent2
{

public:

	__device__ __host__
	Extent2(const int _width = 1, const int _height = 1)
		: width(_width)
		, height(_height)
	{
	}

	__device__ __host__
	Extent2(interval_t& i)
		: width(i.first)
		, height(i.second)
	{
	}

	__device__ __host__
	Extent2(const Extent2& extent)
		: width(extent.width)
		, height(extent.height)
	{
	}

	__device__ __host__
	Extent2(const Pos2& start, const Pos2& end)
		: width(end.x - start.x + 1)
		, height(end.y - start.y + 1)
	{
	}

	__device__ __host__
	int index(const int x, const int y) const
	{
		return y * width + x;
	}

	__device__ __host__
	int index(const Pos2& p) const
	{
		return p.y * width + p.x;
	}

	__device__ __host__
	int checked_index(const int x, const int y = 0) const
	{
		int result = -1;
		if (0 <= x && x < width && 0 <= y && y < height)
		{
			result = y * width + x;
		}
		return result;
	}

	__device__ __host__
	bool in_bounds(const int x, const int y = 0) const
	{
		return x < width && y < height;
	}

	__device__ __host__
	bool in_bounds(const Pos2& p) const
	{
		return p.x < width && p.y < height;
	}

	__device__ __host__
	bool in_bounds_strict(const int x, const int y = 0) const
	{
		return 0 <= x && x < width && 0 <= y && y < height;
	}

	__device__ __host__
	bool in_bounds_strict(const Pos2& p) const
	{
		return 0 <= p.x && p.x < width && 0 <= p.y && p.y < height;
	}

	__device__ __host__
	bool at_border(const int x, const int y = 0) const
	{
		return 0 == x || x == width - 1 || 0 == y ||  y == height-1;
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
	int get_number_of_elems() const
	{
		return get_width() * get_height();
	}

	__device__ __host__
	Pos2 get_previous(const Pos2& p) const
	{
		if (p.x == 0 && p.y > 0)
		{
			return Pos2{ width - 1, p.y - 1 };
		}
		else if (p.x > 0)
		{
			return Pos2{ p.x - 1, p.y };
		}
		else
		{
			return p;
		}
	}

	__device__ __host__
	Extent2 get_extent(const Pos2& p) const
	{
		const int w = p.x + 1;
		const int h = p.y + 1;
		return Extent2{ w, h };
	}

	__device__ __host__
	Extent2& operator=(const Extent2& a)
	{
		width = a.width;
		height = a.height;
		return *this;
	}

	__device__ __host__
	bool operator==(const Extent2& b) const
	{
		return this->width == b.width && this->height == b.height;
	}

	// returns the set of all coordinates (x,y) with 0 <= x < width and 0 <= y < height
	std::vector<std::pair<int, int>> get_coordinates() const
	{
		std::vector<std::pair<int, int>> result;
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				result.push_back(std::pair<int, int>(x, y));
			}
		}
		return result;
	}

private:

	int width;

	int height;

};

inline std::ostream &operator<<(std::ostream& ostr, const Extent2& d)
{
	return ostr << d.get_width() << "," << d.get_height();
}
