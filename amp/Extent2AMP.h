/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include <iostream>
#include "Extent2.h"

class Extent2AMP
{

public:

	Extent2AMP(const int _width = 1, const int _height = 1)
		: width(_width)
		, height(_height)
	{
	}

	Extent2AMP(const Extent2AMP& extent) restrict(cpu, amp)
		: width(extent.width)
		, height(extent.height)
	{
	}

	Extent2AMP(const Extent2& extent)
		: width(extent.get_width())
		, height(extent.get_height())
	{
	}
	
	bool in_bounds_strict(const int x, const int y = 0) const restrict(cpu, amp)
	{
		return 0 <= x && x < width && 0 <= y && y < height;
	}

	//
	//	bool in_bounds_strict(const Pos2& p) const
	//{
	//	return 0 <= p.x && p.x < width && 0 <= p.y && p.y < height;
	//}

	//
	//	bool at_border(const int x, const int y = 0) const
	//{
	//	return 0 == x || x == width - 1 || 0 == y || y == height - 1;
	//}

	//
	//	int get_width() const
	//{
	//	return width;
	//}

	//
	//	int get_height() const
	//{
	//	return height;
	//}

	//
	//	int get_number_of_elems() const
	//{
	//	return get_width() * get_height();
	//}

	//
	//	Pos2 get_previous(const Pos2& p) const
	//{
	//	if (p.x == 0 && p.y > 0)
	//	{
	//		return Pos2{ width - 1, p.y - 1 };
	//	}
	//	else if (p.x > 0)
	//	{
	//		return Pos2{ p.x - 1, p.y };
	//	}
	//	else
	//	{
	//		return p;
	//	}
	//}

	//
	//	Extent2AMP get_extent(const Pos2& p) const
	//{
	//	const int w = p.x + 1;
	//	const int h = p.y + 1;
	//	return Extent2AMP{ w, h };
	//}

	//
	//	Extent2AMP& operator=(const Extent2AMP& a)
	//{
	//	width = a.width;
	//	height = a.height;
	//	return *this;
	//}

	//
	//	bool operator==(const Extent2AMP& b) const
	//{
	//	return this->width == b.width && this->height == b.height;
	//}

	//// returns the set of all coordinates (x,y) with 0 <= x < width and 0 <= y < height
	//std::vector<std::pair<int, int>> get_coordinates() const
	//{
	//	std::vector<std::pair<int, int>> result;
	//	for (int x = 0; x < width; x++)
	//	{
	//		for (int y = 0; y < height; y++)
	//		{
	//			result.push_back(std::pair<int, int>(x, y));
	//		}
	//	}
	//	return result;
	//}

private:

	int width;

	int height;

};

//inline std::ostream &operator<<(std::ostream& ostr, const Extent2AMP& d)
//{
//	return ostr << d.get_width() << "," << d.get_height();
//}
