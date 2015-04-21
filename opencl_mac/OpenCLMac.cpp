/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#define __CL_ENABLE_EXCEPTIONS
#include "cl_1_2.hpp"

#include <iostream>
#include "FunctionCaller.h"
#include "OpenCLUtilities.h"
#include "SimpleTest.h"
#include "SimpleBenchmark.h"
#include "opencl_beispiel.h"
#include "DemoOpenCL.h"
#include "HeatDemoOpenCL.h"
#include "DemoBenchmark.h"
#include "BenchmarkUtilities.h"

using namespace std;

void benchmark_heat_single()
{
	HeatDemoOpenCL demo;
	demo.unset_opengl();
	run(demo, DemoBenchmark{ demo, "Single-OpenCL" });
}

void benchmark2_single()
{
	HeatDemoOpenCL demo;
	run(demo, "Single-OpenCL", get_sizes());
}

int main(int argc, char** argv)
{
	FunctionCaller fc("cuda.exe");
	fc.add("test", []()
	{
		cout << "TEST" << endl;
	}
	);
	fc.add("info", &info_all);
	fc.add("info_p", &info_platforms);
	fc.add("info_d", &info_devices);
	fc.add("info_dd", &info_devices_detailed);
	fc.add("info_tree", &info_tree);
	fc.add("info_t", &info_tree);

	// SimpleTest.h
	fc.add("simple_test", &simple_test_all);
	fc.add("st", &simple_test_all);

	// SimpleBenchmark.h
	fc.add("simple_benchmark", &simple_benchmark_all);
	fc.add("sb", &simple_benchmark_all);
	
	// opencl_beispiel.h
	fc.add("opencl_beispiel", &opencl_beispiel);
	fc.add("bsp", &opencl_beispiel);

	// DemoOpenCL.h
	fc.add("demo_single", &demo_single_opencl);
	fc.add("demo_single_opencl", &demo_single_opencl);
	
	fc.add("benchmark_single", &benchmark_heat_single);

	// call the function
	const int rc = fc.exec(argc, argv);

	// return
	return rc;
}
