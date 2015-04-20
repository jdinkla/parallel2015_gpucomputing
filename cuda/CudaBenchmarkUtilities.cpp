/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "CudaBenchmarkUtilities.h"
#include <iostream>
#include "CudaUtilities.h"
#include <cmath>
#include <vector>

using namespace std;

blocks_t get_naive_blocks()
{
	blocks_t result{
		dim3(32, 1), dim3(30, 1), dim3(31, 1), dim3(33, 1), dim3(34, 1),
		dim3(32, 2), dim3(30, 2), dim3(31, 2), dim3(33, 2), dim3(34, 2),
		dim3(32, 3), dim3(30, 3), dim3(31, 3), dim3(33, 3), dim3(34, 3),
		dim3(32, 4), dim3(30, 4), dim3(31, 4), dim3(33, 4), dim3(34, 4),
		dim3(32, 5), dim3(30, 5), dim3(31, 5), dim3(33, 5), dim3(34, 5)
	};
	return result;
}

blocks_t get_blocks()
{
	blocks_t result{ 
		dim3(32, 1), dim3(32, 2), dim3(32, 3), dim3(32, 4),
		dim3(32, 5), dim3(32, 6), dim3(64, 1), dim3(64, 2),
		dim3(64, 3), dim3(64, 4), dim3(96, 1), dim3(96, 2),
		dim3(96, 3), dim3(128, 1), dim3(128, 2), dim3(128, 3),
		dim3(128, 4), dim3(192, 1), dim3(192, 2), dim3(192, 3),
		dim3(256, 1), dim3(256, 2), dim3(256, 3), dim3(256, 4),
		dim3(512, 1), dim3(512, 2), dim3(1024, 1)
	};
	return result;
}

blocks_t get_special_blocks()
{
	blocks_t result{
		dim3(128, 1),
		dim3(256, 1),
		dim3(192, 1),
		dim3(512, 1),
		dim3(96, 1),
		dim3(1024, 1),
		dim3(64, 1)
	};
	return result;
}

size_t get_max_size_2d(const bool silent)
{
	const size_t free = CUDA::get_free_device_mem();
	if (!silent)
	{
		cout << "free memory " << free << endl;
	}
	const size_t free2 = (size_t) (free * 0.65);	// Sicherheit
	const size_t free3 = free2 / 2;					// zwei buffer
	const size_t free4 = free3 / 4;					// int/float brauchen 4 bytes
	const size_t squared = (size_t) sqrt(free4);
	if (!silent)
	{
		cout << "Using sizeX=sizeY=" << squared << endl;
	}
	return squared;
}
