/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

template <class Framework, class Data>
struct Partition
{
	int partition_id;
	Framework framework;
	Data data;
};

#ifdef COMPILING_FOR_CUDA
struct CudaPartition1D
{
	int partition_id;
	int device_id;
	cudaStream_t stream;				// not reusable for OpenCL, AMP
	int start;							// not reusable for 2D, 3D, etc.
	int end;
};
#endif
