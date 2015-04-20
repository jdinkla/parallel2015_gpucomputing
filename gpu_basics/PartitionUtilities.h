/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <iostream>
#include <vector>

typedef std::pair<int, int> interval_t;
typedef std::vector<interval_t> partition_t;

// divide the intervall from start top end into num_parts parts
partition_t partition(const int start, const int end, const int num_parts);

// divide the intervall from start top end into num_parts parts with overlap
// the overlap exclusive, i.e. is simply subtracted from the start and added to the end
partition_t partition(const int start, const int end, const int num_parts, const int overlap);


typedef struct
{
	interval_t with_overlap;
	interval_t without_overlap;
} pa_t;

inline std::ostream &operator<<(std::ostream& ostr, const interval_t& d)
{
	return ostr << "(" << d.first << "," << d.second << ")";
}
