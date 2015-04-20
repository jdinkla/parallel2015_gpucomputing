/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "CudaExecConfig.h"
#include "Utilities.h"

CudaExecConfig::CudaExecConfig(const Extent2& _extent, const dim3 _block, const int _shared_mem, cudaStream_t _stream)
	: block(_block)
	, grid(ceiling_div(extent.get_width(), _block.x), ceiling_div(extent.get_height(), _block.y), 1)
	, shared_mem(_shared_mem)
	, stream(_stream)
	, extent(_extent)
{
}

CudaExecConfig::CudaExecConfig(const int num_elems, const dim3 _block /*= dim3(128, 1, 1)*/, const int _shared_mem /*= 0*/, cudaStream_t _stream /*= 0*/)
	: block(_block)
	, grid(ceiling_div(extent.get_width(), _block.x), ceiling_div(extent.get_height(), _block.y), 1)
	, shared_mem(_shared_mem)
	, stream(_stream)
	, extent(Extent2(num_elems, 1))
{
}
