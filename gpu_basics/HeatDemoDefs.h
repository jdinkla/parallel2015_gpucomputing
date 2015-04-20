/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#ifdef _DEBUG
const int iterations = 25;
#else
const int iterations = 100;
#endif

const int benchmark_iterations_cpu = 100;

#ifdef _DEBUG
const int benchmark_iterations_gpu = 25;
#else
const int benchmark_iterations_gpu = 1000;
#endif

const int default_size_xy = 1024 * 10;
const int default_width = default_size_xy;
const int default_height = default_size_xy;

const int small_size_xy = 1024;
const int small_width = small_size_xy;
const int small_height = small_size_xy;

const int num_heat_sources = 100;
const float ct = 0.25f;
const float max_heat = 100.0f;
const float ratio_of_heat_sources = 0.0001f;

