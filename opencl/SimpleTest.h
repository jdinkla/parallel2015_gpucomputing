/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include "cl_1_2.hpp"
#include "UnpinnedBuffer.h"

// initializes the buffer with i*i
void init(UnpinnedBuffer<float>& h);

// verifies the buffer
bool verify(UnpinnedBuffer<float>& h);

// call the square.cl kernel 
void simple_test(UnpinnedBuffer<float>& h,
	cl::Platform& platform,
	cl::Device& device,
	cl::Program::Sources& sources);

void simple_test_all();

void simple_test_laptop();

void simple_test_multi_gpu();

