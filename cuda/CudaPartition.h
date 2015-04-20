/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "GPU.h"
#include "GPUs.h"
#include "XExtent2.h"
#include "Partition.h"

// the partitions
struct framework
{
	GPU gpu;
};

struct data
{
	XExtent2 xext;
};

using CudaPartition = Partition < framework, data > ;

std::vector<CudaPartition> calc_partitions(GPUs& gpus, const Extent2& extent);


