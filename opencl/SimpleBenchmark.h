/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include "cl_1_2.hpp"
#include "UnpinnedBuffer.h"

// call the square.cl kernel 
void simple_benchmark(UnpinnedBuffer<float>& h,
	cl::Platform& platform,
	cl::Device& device,
	cl::Program::Sources& sources);

void simple_benchmark_all();
