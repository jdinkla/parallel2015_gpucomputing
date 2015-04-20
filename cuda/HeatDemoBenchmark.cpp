/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "HeatDemoBenchmark.h"
#include "DemoBenchmark.h"
#include "HeatDemoCPU.h"
#include "HeatDemoSingleGPU.h"
#include "HeatDemoMultiGPU.h"
#include "SequentialScheduler.h"
#include "AsyncScheduler.h"
#include "ThreadScheduler.h"
#include "DemoBenchmark.h"
#include "CudaUtilities.h"

const int device = 0;

void heat_demo_benchmark()
{
	cudaSetDevice(device);
	CUDA::check("cudaSetDevice");
	HeatDemoSingleGPU demo;
	string name = "Single-CUDA";
	DemoBenchmark bench{ demo, name };
	bench.iterations = benchmark_iterations_gpu;
	run(demo, bench);
}

void heat_multi_demo_benchmark()
{
	HeatDemoMultiGPU demo;
	demo.unset_opengl();
	string name = "Multi-CUDA";
	DemoBenchmark bench{ demo, name };
	bench.iterations = benchmark_iterations_gpu;
	run(demo, bench);
}

void heat_multi_demo_benchmark_seq()
{
	HeatDemoMultiGPU demo;
	demo.unset_opengl();
	demo.set_scheduler(new SequentialScheduler<CudaPartition>());
	string name = "Multi-CUDA-Seq";
	DemoBenchmark bench{ demo, name };
	bench.iterations = benchmark_iterations_gpu;
	run(demo, bench);
}

void heat_multi_demo_benchmark_async()
{
	HeatDemoMultiGPU demo;
	demo.unset_opengl();
	demo.set_scheduler(new AsyncScheduler<CudaPartition>());
	string name = "Multi-CUDA-Async";
	DemoBenchmark bench{ demo, name };
	bench.iterations = benchmark_iterations_gpu;
	run(demo, bench);
}

void heat_multi_demo_benchmark_threads()
{
	HeatDemoMultiGPU demo;
	demo.unset_opengl();
	demo.set_scheduler(new ThreadScheduler<CudaPartition>());
	string name = "Multi-CUDA-Threads";
	DemoBenchmark bench{ demo, name };
	bench.iterations = benchmark_iterations_gpu;
	run(demo, bench);
}

void heat_demo_benchmark_cpu()
{
	HeatDemoCPU demo;
	demo.unset_opengl();
	demo.set_parallel(false);
	string name = "Single-CPU";
	DemoBenchmark bench{ demo, name };
	bench.iterations = benchmark_iterations_cpu / 5;
	run(demo, bench);
}

void heat_multi_demo_benchmark_cpu()
{
	HeatDemoCPU demo;
	demo.unset_opengl();
	demo.set_parallel(true);
	string name = "Multi-CPU-Async";
	DemoBenchmark bench{ demo, name };
	bench.iterations = benchmark_iterations_cpu;
	run(demo, bench);
}

void heat_multi_single_vs_multi()
{
	bool is_first = true;
	string name_s = "single GPU";
	string name_m = "multi GPU async";

	vector<int> sizes;
	for (int i = 1; i < 30; i++)
	{
		sizes.push_back(i * 1024);
	}

	for (auto& sz : sizes)
	{
		cudaSetDevice(device);
		CUDA::check("cudaSetDevice");

		try
		{
			HeatDemoSingleGPU demo_s;
			stringstream str_s;
			str_s << name_s << ";" << sz;
			DemoBenchmark bench_s{ demo_s, str_s.str(), sz, sz };
			bench_s.iterations = benchmark_iterations_gpu;
			bench_s.print_header = is_first;
			bench_s.save_results = false;
			run(demo_s, bench_s);
		}
		catch (std::exception & e)
		{
		}

		is_first = false;

		try
		{
			HeatDemoMultiGPU demo_m;
			demo_m.unset_opengl();
			demo_m.set_scheduler(new AsyncScheduler<CudaPartition>());
			stringstream str_m;
			str_m << name_m << ";" << sz;
			DemoBenchmark bench_m{ demo_m, str_m.str(), sz, sz };
			bench_m.iterations = benchmark_iterations_gpu;
			bench_m.print_header = is_first;
			bench_m.save_results = false;
			run(demo_m, bench_m);
		}
		catch (std::exception & e)
		{
		}
	}

}

void run_demo(IDemo& demo, vector<int>& sizes)
{
	bool is_first = true;
	for (auto& sz : sizes)
	{
		try
		{
			DemoBenchmark bench{ demo, "???", sz, sz };
			bench.iterations = benchmark_iterations_gpu;
			bench.print_header = is_first;
			bench.save_results = false;
			run(demo, bench);
		}
		catch (std::exception & e)
		{
			cerr << "ERROR " << e.what() << endl;
		}
		is_first = false;
	}
}