/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Extent2.h"
#include "Pos2.h"
#include <iostream>

// A Region2 is a part of an Extent2.

class Region2 
{

public:

	__device__ __host__
	Region2(const Extent2& _local, const Pos2& _offset)
		: local(_local)
		, offset(_offset)
	{
	}

	__device__ __host__
	Extent2 get_extent() const
	{
		return local;
	}

	__device__ __host__
	Pos2 get_offset() const
	{
		return offset;
	}

private:

	Extent2 local;
	Pos2 offset;
};

inline std::ostream &operator<<(std::ostream& ostr, const Region2& r)
{
	return ostr << r.get_extent() << "," << r.get_offset();
}
