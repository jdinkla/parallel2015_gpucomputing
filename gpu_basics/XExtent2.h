/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include <iostream>
//#include <vector>
#include "Region2.h"
#include "Extent2.h"

// offset is the "offset" of the region within the global extent
// overlap is the number of overlapping regions at the border. 
// for example: for a global extent [0, 7] and an offset 2 and overlap 1 there is [2, 

class XExtent2
{

public:

	__device__ __host__
	XExtent2(
		const Extent2& _global_extent,
		const Extent2& _local_extent_with_overlap,
		const Pos2& _offset_with_overlap,
		const Extent2& _local_extent_without_overlap,
		const Pos2& _offset_without_overlap)
		: global_extent(_global_extent)
		, outer(_local_extent_with_overlap, _offset_with_overlap)
		, inner(_local_extent_without_overlap, _offset_without_overlap)
		, outer_offset(_offset_with_overlap)
		, inner_offset(_offset_without_overlap)
	{
	}

	__device__ __host__
	Extent2 get_global_extent() const
	{
		return global_extent;
	}

	__device__ __host__
	Region2 get_outer() const
	{
		return outer;
	}

	__device__ __host__
	Region2 get_inner() const
	{
		return inner;
	}
	
	__device__ __host__
	Pos2 get_outer_offset() const
	{
		return outer_offset;
	}

	__device__ __host__
	Pos2 get_inner_offset() const
	{
		return inner_offset;
	}

	__device__ __host__
	bool has_low_overlap() const
	{
		return outer_offset != inner_offset;
	}

	__device__ __host__
	Region2 get_low_overlap() const
	{
		if (has_low_overlap())
		{
			Pos2 offset_start = outer_offset;
			Pos2 offset_end = global_extent.get_previous(inner_offset);
			Extent2 ext = Extent2(offset_start, offset_end);
			return Region2{ ext, outer_offset };
		}
		else
		{
			return Region2{ Extent2{ 0, 0 }, Pos2{ 0, 0 } };
		}
	}

	__device__ __host__
	bool has_high_overlap() const
	{
		if (outer_offset == inner_offset)
		{
			// no low overlap and has to have high, if extents are different
			return !(outer.get_extent() == inner.get_extent());
		}
		else
		{
			// has low overlap
			// outer = low + inner + hi  <=> hi = outer - low - inner
			Extent2 out = outer.get_extent();
			Extent2 low = get_low_overlap().get_extent();
			Extent2 inn = inner.get_extent();
			const int hi = out.get_height() - low.get_height() - inn.get_height();
			return hi > 0;
		}
	}

	__device__ __host__
	Region2 get_high_overlap() const
	{
		if (has_high_overlap())
		{
			Pos2 offset_start = inner_offset;
			offset_start.y += inner.get_extent().get_height();

			Pos2 offset_end = outer_offset;
			offset_end.x = outer.get_extent().get_width() - 1;
			offset_end.y += outer.get_extent().get_height() - 1;

			Extent2 ext = Extent2(offset_start, offset_end);
			return Region2{ ext, offset_start };
		}
		else
		{
			return Region2{ Extent2{ 0, 0 }, Pos2{ 0, 0 } };
		}
	}

	__device__ __host__
	Region2 get_low_inner() const
	{
		Region2 low = get_low_overlap();
		Extent2 ext = low.get_extent();
		return Region2{ ext, inner_offset };
	}

	__device__ __host__
	Region2 get_high_inner() const
	{
		Region2 hi = get_high_overlap();
		Extent2 ext = hi.get_extent();
		Pos2 pos = hi.get_offset();
		pos.y -= ext.get_height();
		return Region2{ ext, pos };
	}

	//__device__ __host__
	//bool has_overlap() const
	//{
	//	return !(local_with_overlap.get_local() == local_without_overlap.get_local());
	//}

	//Pos2 get_offset_of_high_overlap_in_global() const
	//{
	//	const Pos2 last_without = local_without_overlap.get_last_pos_in_global();
	//	return Pos2{ 0, last_without.y + 1 };
	//}

	Pos2 get_offset_in_local() const
	{
		return inner_offset - outer_offset;
	}

	//int get_offset_in_local_index() const
	//{
	//	return local_with_overlap.get_local().index(get_offset_in_local());
	//}

	Pos2 transform_global_to_outer(const Pos2& pos) const
	{
		return pos - outer_offset;
	}

	// Returns the start of the "high inner" region
	int get_high_inner_start_index() const
	{
		Extent2 ext = get_outer().get_extent();
		Region2 hii = get_high_inner();
		Pos2 global_offset = hii.get_offset();
		Pos2 outer_offset = transform_global_to_outer(global_offset);
		const int idx = ext.index(outer_offset);
		return idx;
	}

	// Returns the start of the "low inner" region
	int get_low_inner_start_index() const
	{
		Extent2 ext = get_outer().get_extent();
		Region2 loi = get_low_inner();
		Pos2 global_offset = loi.get_offset();
		Pos2 outer_offset = transform_global_to_outer(global_offset);
		const int idx = ext.index(outer_offset);
		return idx;
	}

	// Returns the start of the "high overlap" region
	int get_high_overlap_start_index() const
	{
		Extent2 ext = get_outer().get_extent();
		Region2 hio = get_high_overlap();
		Pos2 global_offset2 = hio.get_offset();
		Pos2 outer_offset2 = transform_global_to_outer(global_offset2);
		const int idx = ext.index(outer_offset2);
		return idx;
	}

private:

	// the size of the global "surroundings"
	Extent2 global_extent;

	// the size of this partition with overlap
	Region2 outer;

	// the size of this partition without overlap
	Region2 inner;

	Pos2 outer_offset;

	Pos2 inner_offset;

};

//std::vector<Partition2> create_partitions(Extent2& _extent, const int _number_of_partitions, const int _overlap);

//inline std::ostream &operator<<(std::ostream& ostr, const Partition2& d)
//{
//	return ostr << d.global_extent << ", " << d.local_with_overlap << ", " << d.local_without_overlap << ", " << d.extent;
//}
