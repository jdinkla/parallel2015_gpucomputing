/*
 * Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#pragma once

#include <cuda_runtime_api.h>
#include "Extent2.h"

class CudaExecConfig
{

public:
	
	CudaExecConfig(const Extent2& _extent, const dim3 _block = dim3(128, 1, 1), const int _shared_mem = 0, cudaStream_t _stream = 0);

	CudaExecConfig(const int num_elems, const dim3 _block = dim3(128, 1, 1), const int _shared_mem = 0, cudaStream_t _stream = 0);

	dim3 get_grid() const
	{
		return grid;
	}

	dim3 get_block() const
	{
		return block;
	}

	int get_shared_mem() const
	{
		return shared_mem;
	}

	cudaStream_t get_stream() const
	{
		return stream;
	}

	void set_stream(cudaStream_t _stream)
	{
		stream = _stream;
	}

	Extent2 get_extent() const
	{
		return extent;
	}

private:

	Extent2 extent;
	dim3 grid;
	dim3 block;
	int shared_mem;
	cudaStream_t stream;

};