/*
 * Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "Defs.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <assert.h>

#ifdef MAC
#include "../gpu_basics/Timer.h"
#else
#include "Timer.h"
#endif
#include "FunctionCaller.h"
#include "GPUs.h"
#include "CudaBenchmarkUtilities.h"
#include "Utilities.h"
#include "CudaUtilities.h"
#include "ColorUtilities.h"
#include "HeatDemoBenchmark.h"
#include "PartitionUtilities.h"
#include "Region2.h"
#include "DemoSingleGPU.h"
#include "DemoMultiGPU.h"
#include "XExtent2.h"
#include "HeatDemoSingleGPU.h"
#include "HeatDemoMultiGPU.h"
#include "BenchmarkUtilities.h"
#include "DemoBenchmark.h"
#include "HeatDemoCPU.h"

using namespace std;

void x_extent() 
{
	Extent2 e1{ Pos2{ 0, 2 }, Pos2{ 13, 3 } };
	assert(e1.get_width() == 14);
	assert(e1.get_height() == 2);

	Extent2 g(4, 8);
	Extent2 lo(4, 5);
	Extent2 li(4, 3);
	Pos2 oo(0, 2);
	Pos2 oi(0, 3);
	XExtent2 se{ g, lo, oo, li, oi };

	Region2 low = se.get_low_overlap();
	bool b = low.get_extent() == Extent2{ 4, 1 };
	assert(b);
	b = low.get_offset() == oo;
	assert(b);

	Region2 hig = se.get_high_overlap();
	b = hig.get_extent() == Extent2{ 4, 1 };
	assert(b);
	b = hig.get_offset() == Pos2{0,6};
	assert(b);

	Region2 loi = se.get_low_inner();
	b = loi.get_extent() == Extent2{ 4, 1 };
	assert(b);
	b = loi.get_offset() == Pos2{ 0, 3 };
	assert(b);

	Region2 hii = se.get_high_inner();
	b = hii.get_extent() == Extent2{ 4, 1 };
	assert(b);
	b = hii.get_offset() == Pos2{ 0, 5 };
	assert(b);

	XExtent2 se1{ Extent2{ 706, 504 }, Extent2{ 706, 253 }, Pos2{ 0, 0 }, Extent2{ 706, 252 }, Pos2{ 0, 0 } };
	XExtent2 se2{ Extent2{ 706, 504 }, Extent2{ 706, 253 }, Pos2{ 0, 251 }, Extent2{ 706, 252 }, Pos2{ 0, 252 } };
	//cout << "1:" << se1.has_low_overlap() << endl;
	//cout << "2:" << se1.has_high_overlap() << endl;
	//cout << "3:" << se2.has_low_overlap() << endl;
	//cout << "4:" << se2.has_high_overlap() << endl;
	assert(!se1.has_low_overlap());
	assert(se1.has_high_overlap());
	assert(se2.has_low_overlap());
	assert(!se2.has_high_overlap());
}

void hsl()
{
	for (float i = 0; i <= 1.0f; i += 0.1f)
	{
		uchar4 rgb = hsl_to_rgb(i, 0.5f, 0.5f);
		cout << setw(2) << i << " = " << rgb << endl;
	}
}

void x()
{
	GPUs g;
	cout << "n=" << g.get_p2p_gpus().size() << endl;
}

void mem()
{
	int count = -1;
	cudaGetDeviceCount(&count);
	for (int d = 0; d < count; d++)
	{
		cudaSetDevice(d);
		cout << "device " << d << endl;
		size_t sz = get_max_size_2d();
	}
}

void benchmark2_single()
{
	HeatDemoSingleGPU demo;
	run(demo, "Single-CUDA", get_sizes());
}

void benchmark2_multi()
{
	HeatDemoMultiGPU demo;
	demo.unset_opengl();
	run(demo, "Multi-CUDA", get_sizes());
}

void benchmark2_single_cpu()
{
	HeatDemoCPU demo;
	demo.set_parallel(false);
	run(demo, "Single-Thread", get_sizes());
}

void benchmark2_multi_cpu()
{
	HeatDemoCPU demo;
	demo.set_parallel(true);
	demo.unset_opengl();
	run(demo, "Multi-Thread", get_sizes());
}

int main(int argc, char** argv)
{
	FunctionCaller fc("cuda.exe");
	fc.add("test", []()
	{
		cout << "TEST" << endl;
	});
	fc.add("x", &x);
	fc.add("check_chrono", &check_chrono);
	fc.add("mem", &mem);
	fc.add("hsl", &hsl);
	fc.add("xext", &x_extent);

	// HeatDemoSingleGPUStandalone.h
	fc.add("demo_single", &demo_single);

	// HeatDemoMultiGPUStandalone.h
	fc.add("demo_multi", &demo_multi);

	// heat benchmark from HeatDemoBenchmark.h
	fc.add("benchmark_single", &heat_demo_benchmark);
	fc.add("benchmark_single_gpu", &heat_demo_benchmark);
	fc.add("benchmark_multi_gpu", &heat_multi_demo_benchmark);
	fc.add("benchmark_multi_seq", &heat_multi_demo_benchmark_seq);
	fc.add("benchmark_multi", &heat_multi_demo_benchmark_async);
	fc.add("benchmark_multi_async", &heat_multi_demo_benchmark_async);
	fc.add("benchmark_multi_threads", &heat_multi_demo_benchmark_threads);
	fc.add("benchmark_single_cpu", &heat_demo_benchmark_cpu);
	fc.add("benchmark_multi_cpu", &heat_multi_demo_benchmark_cpu);
	fc.add("benchmark_vs", &heat_multi_single_vs_multi);

	fc.add("benchmark2_single", &benchmark2_single);
	fc.add("benchmark2_multi", &benchmark2_multi);

	fc.add("benchmark2_single_cpu", &benchmark2_single_cpu);
	fc.add("benchmark2_multi_cpu", &benchmark2_multi_cpu);
	
	// call the function
	const int rc = fc.exec(argc, argv);

	// a device reset is needed by the profiler
	CUDA::reset_all();

	return rc;
}