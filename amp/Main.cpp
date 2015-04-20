/*
 * Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#include <iostream>
#include "Beispiele.h"
#include "FunctionCaller.h"
#include "AMPUtilities.h"
#include "DemoAMP.h"
#include "DemoBenchmark.h"
#include "HeatDemoAMP.h"
#include "BenchmarkUtilities.h"

using namespace std;

void benchmark_single()
{
	HeatDemoAMP demo;
	run(demo, DemoBenchmark{ demo, "Single-AMP" });
}

void benchmark2_single()
{
	HeatDemoAMP demo;
	run(demo, "Single-AMP", get_sizes());
}

int main(int argc, char** argv)
{
	FunctionCaller fc("cuda.exe");

	// Beispiele.h
	fc.add("amp", &amp_map_beispiel);

	// AMPUtilities.h
	fc.add("info", &amp_info);

	// DemoAMP.h"
	fc.add("demo_single", &demo_single_amp);
	fc.add("demo_single_amp", &demo_single_amp);
	fc.add("demo_single_all", &demo_single_amp_all_devices);
	fc.add("demo_single_amp_all", &demo_single_amp_all_devices);

	// Benchmark
	fc.add("benchmark_single", &benchmark_single);
	fc.add("benchmark2_single", &benchmark2_single);

	// call the function
	const int rc = fc.exec(argc, argv);

	// return
	return rc;
}