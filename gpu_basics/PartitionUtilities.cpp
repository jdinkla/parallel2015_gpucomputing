/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "PartitionUtilities.h"
#include <algorithm>

partition_t partition(const int start, const int end, const int num_parts)
{
	partition_t result;
	const int size_overall = end - start + 1;
	const int size_part = (size_overall + num_parts - 1) / num_parts;
	int i = start;
	while (i <= end)
	{
		interval_t p(i, std::min(i + size_part - 1, end));
		result.push_back(p);
		i += size_part;
	}
	return result;
}

partition_t partition(const int start, const int end, const int num_parts, const int overlap)
{
	partition_t result;
	partition_t ps = partition(start, end, num_parts);

	for (auto& p : ps)
	{
		interval_t i(std::max(p.first - overlap, start), std::min(p.second + overlap, end));
		result.push_back(i);
	}
	return result;
}
