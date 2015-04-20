/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "XExtent2.h"
#include "Partition.h"
#include <amp.h>

struct amp_framework
{
	concurrency::accelerator gpu;
};

struct amp_data
{
	XExtent2 xext;
};

using AMPPartition = Partition < amp_framework, amp_data >;

using Accels = std::vector < concurrency::accelerator > ;

std::vector<AMPPartition> calc_AMP_partitions(Accels& gpus, const Extent2& extent);


