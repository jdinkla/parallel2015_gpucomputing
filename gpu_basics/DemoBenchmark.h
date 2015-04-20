/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "IDemo.h"
#include <string>
#include "HeatDemoDefs.h"

class DemoBenchmark
{

public:

	DemoBenchmark(IDemo& _demo, std::string _name, const int _width = default_width, const int _height = default_height);

	void operator()() const;

	int iterations = benchmark_iterations_gpu;

	const int width;
	const int height;
	bool print_all = false;
	bool save_results = true;
	bool render = false;
	bool print_header = true;

private:

	IDemo& demo;

	std::string name;

};

void run(IDemo& demo, const DemoBenchmark& bench);
void run(IDemo& demo, std::string name, std::vector<int> sizes);
