/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include <iostream>

// A Position in an Extent2. 
class Pos2
{

public:

	__device__ __host__
	Pos2(const int _x = 0, const int _y = 0)
		: x(_x)
		, y(_y)
	{
	}

	__device__ __host__
	Pos2(const Pos2& pos)
		: x(pos.x)
		, y(pos.y)
	{
	}

	__device__ __host__
	bool operator==(const Pos2& b) const
	{
		return this->x == b.x && this->y == b.y;
	}

	__device__ __host__
	bool operator!=(const Pos2& b) const
	{
		return this->x != b.x || this->y != b.y;
	}

	// data
	int x;

	int y;

};

__device__ __host__ inline
Pos2 operator+(const Pos2& l, const Pos2& r) 
{
	return Pos2 {l.x + r.x, l.y + r.y};
}

__device__ __host__ inline
Pos2 operator-(const Pos2& l, const Pos2& r) 
{
	return Pos2 {l.x - r.x, l.y - r.y};
}

inline std::ostream &operator<<(std::ostream& ostr, const Pos2& d)
{
	return ostr << "(" << d.x << "," << d.y << ")";
}
