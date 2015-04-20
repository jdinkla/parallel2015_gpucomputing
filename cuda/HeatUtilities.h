/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "HostBuffer.h"
#include "UnpinnedBuffer.h"
#include <memory>
#include <random>

#ifdef COMPILE_FOR_CUDA
#include "DeviceBuffer.h"
#include "ManagedBuffer.h"
#include "PinnedBuffer.h"

#endif

// used by the standalone versions
//void generate_heat(float *ptr, const Extent2& ext, const int num_elems, const int seed = 1234);
//
//void generate_heat(HostBuffer<float>& buf, const int num_sources, const int seed = 1234);

//void generate_heat(float *ptr, const int num_elems, const int seed = 1234);
//
//
//void generate_heat(std::shared_ptr<HostBuffer<float>> buf, const int num_elems, const int seed = 1234);
//
//UnpinnedBuffer<float>* create_heat_host(const Extent2& ext, const int num_points, const int seed = 1234);

//void do_something_with_the_data(HostBuffer<float>& host);

#ifdef COMPILE_FOR_CUDA

//DeviceBuffer<float>* create_heat(const Extent2& ext, const int num_points, const int seed = 1234);

ManagedBuffer<uchar4>* get_color_buffer();

#endif


