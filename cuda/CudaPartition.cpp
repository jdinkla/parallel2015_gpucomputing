/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/


#include "CudaPartition.h"

using namespace std;

vector<CudaPartition> calc_partitions(GPUs& gpus, const Extent2& extent)
{
	vector<CudaPartition> partitions;

	const int num_gpus = gpus.get_number_of_gpus();

	const int width = extent.get_width();
	const int height = extent.get_height();

	partition_t ps1 = partition(0, height - 1, num_gpus);
	partition_t ps2 = partition(0, height - 1, num_gpus, 1);

	partitions.clear();
	int i = 0;
	for (auto& gpu : gpus)
	{
		int y_low, y_hi, y_len;

		y_low = ps2[i].first;
		y_hi = ps2[i].second;
		y_len = y_hi - y_low + 1;

		Extent2 local_extent_with_overlap(width, y_len);
		Pos2 offset_with_overlap(0, y_low);

		y_low = ps1[i].first;
		y_hi = ps1[i].second;
		y_len = y_hi - y_low + 1;

		Extent2 local_extent_without_overlap(width, y_len);
		Pos2  offset_without_overlap(0, y_low);

		framework f{ gpu };
		data d{ XExtent2(extent, local_extent_with_overlap, offset_with_overlap, local_extent_without_overlap, offset_without_overlap) };

		CudaPartition p{ i, f, d };
		partitions.push_back(p);
		i++;
	}
	return partitions;
}
