/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "Defs.h"
#include "DemoBenchmark.h"
#include "Utilities.h"
#include "Extent2.h"
#include "Timer.h"
#include "FileUtilities.h"
#include <sstream>
#include "HeatDemoDefs.h"
#include "ConstantHeatSource.h"

using namespace std;
using TheTimer = Timer;

const int constant_step = 97;
const int constant_mod = 100;

void run(IDemo& demo, const DemoBenchmark& bench)
{
	demo.set_heat_source(ConstantHeatSource::make_shared(constant_step, constant_mod));
	try
	{
		bench();
	}
	catch (std::exception& e)
	{
		cerr << "ERROR " << e.what() << endl;
	}
	catch (...)
	{
		cerr << "Unknown ERROR" << endl;
	}
}

void run(IDemo& demo, string name, vector<int> sizes)
{
	demo.set_heat_source(ConstantHeatSource::make_shared(constant_step, constant_mod));
	bool is_first = true;
	for (auto sz : sizes)
	{
		try
		{
			DemoBenchmark bench(demo, name, sz, sz);
			bench.print_header = is_first;
			bench();
			is_first = false;
		}
		catch (std::exception& e)
		{
			cerr << "ERROR " << e.what() << endl;
		}
		catch (...)
		{
			cerr << "Unknown ERROR" << endl;
		}
	}
}

DemoBenchmark::DemoBenchmark(IDemo& _demo, std::string _name, const int _width, const int _height)
	: demo(_demo)
	, name(_name)
	, width(_width)
	, height(_height)
{
}

void DemoBenchmark::operator()() const
{
	// create image buffer (done by OpenGL if run as app)
	Extent2 ext(width, height);
	demo.init(width, height);

	float sum_loop, sum_update;
	sum_loop = sum_update = 0.0f;

	if (print_header)
	{
		print_csv("OS", "test", "stage", "width", "height", "i", "loop", "update", "all");
	}

	TheTimer time_all;
	time_all.start();
	// loop
	for (int i = 0; i < iterations; i++)
	{
		TheTimer time_loop, time_update;

		time_loop.start();

		time_update.start();
		demo.update();
		time_update.stop();

		time_loop.stop();

		sum_loop += time_loop.delta();
		sum_update += time_update.delta();

		if (print_all)
		{
			print_csv(OS_NAME, name, "loop", width, height, i, time_loop.delta(), time_update.delta());
		}
	}

	demo.synchronize();

	time_all.stop();

	// print durations
	print_csv(OS_NAME, name, "final", width, height, iterations, sum_loop, sum_update, time_all.delta());

	if (save_results)
	{
		// save result as file
		stringstream filename;
		filename << name << ".bmp";
		demo.save(filename.str());
	}

	demo.cleanup();
	demo.shutdown();
}


