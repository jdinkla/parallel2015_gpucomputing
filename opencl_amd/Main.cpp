/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <iostream>
#include "FunctionCaller.h"
//#include "OpenCLUtilities.h"
//#include "SimpleTest.h"
//#include "SimpleBenchmark.h"
#include "bolt_beispiel.h"

using namespace std;

int main(int argc, char** argv)
{
	FunctionCaller fc("cuda.exe");
	fc.add("test", []()
	{
		cout << "TEST" << endl;
	}
	);
	//fc.add("info", &info_all);
	//fc.add("info_p", &info_platforms);
	//fc.add("info_d", &info_devices);
	//fc.add("info_dd", &info_devices_detailed);
	//fc.add("info_tree", &info_tree);
	//fc.add("info_t", &info_tree);

	//// SimpleTest.h
	//fc.add("simple_test", &simple_test_all);
	//fc.add("st", &simple_test_all);

	//// SimpleBenchmark.h
	//fc.add("simple_benchmark", &simple_benchmark_all);
	//fc.add("sb", &simple_benchmark_all);

	//// opencl_beispiel.h
	//fc.add("opencl_beispiel", &opencl_beispiel);
	//fc.add("bsp", &opencl_beispiel);

	// bolt_beispiel.h
	fc.add("bolt_beispiel", &bolt_beispiel);
	fc.add("bolt", &bolt_beispiel);
	
	// call the function
	const int rc = fc.exec(argc, argv);

	// return
	return rc;
}
