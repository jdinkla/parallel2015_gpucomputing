/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#define __CL_ENABLE_EXCEPTIONS

#ifdef WINDOWS
#include <omp.h>
#endif

#include "SimpleBenchmark.h"
#include "cl_1_2.hpp"
#include "UnpinnedBuffer.h"
#include <algorithm>
#include "OpenCLUtilities.h"
#include "FileUtilities.h"
#include <iostream>
#include "Timer.h"

using namespace std;

static const char* kernel_filename = "../opencl/cl/square.cl";

// initializes the buffer with i*i
void init2(UnpinnedBuffer<float>& h)
{
	fill(h.begin(), h.end(), 4);
}

// verifies the buffer
bool verify2(UnpinnedBuffer<float>& h)
{
	return std::all_of(h.begin(), h.end(), [](int v) { return v == 4*4; });
}

void host_seq(UnpinnedBuffer<float>& h)
{
	const Extent2 ext = h.get_extent();
	const int width = ext.get_width();
	const int height = ext.get_height();
	float* ptr = h.get_ptr();
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			const int idx = ext.index(x, y);
			const float value = ptr[idx];
			ptr[idx] = value * value;
		}
	}
}

void host_par(UnpinnedBuffer<float>& h)
{
	const Extent2 ext = h.get_extent();
	const int width = ext.get_width();
	const int height = ext.get_height();
	float* ptr = h.get_ptr();
#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			const int idx = ext.index(x, y);
			const float value = ptr[idx];
			ptr[idx] = value * value;
		}
	}
}

// call the square.cl kernel 
void simple_benchmark(UnpinnedBuffer<float>& h,
	cl::Platform& platform,
	cl::Device& device,
	cl::Program::Sources& sources)
{
	Timer timer;

	cl::Context context = get_context(platform, device);
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// program

	cl::Program program = cl::Program(context, sources);
	program.build({ device });
	cl::Kernel square_kernel(program, "square");

	// kernel & buffer
	cl::Buffer buffer(context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		h.get_size_in_bytes(), h.get_ptr());

	square_kernel.setArg(0, buffer);

	// fill the queue
	cl::Event event, event_kernel, event_all1, event_all2;

	/*
	queue.enqueueWriteBuffer(
	buffer,
	false,
	0,
	h.get_size_in_bytes(),
	h.get_ptr(),
	NULL);
	*/
	timer.start();
	queue.enqueueNDRangeKernel(
		square_kernel,
		cl::NullRange,
#if 0
		cl::NDRange(h.get_extent().get_number_of_elems()),
#else
		cl::NDRange(h.get_extent().get_width(), h.get_extent().get_height()),
#endif
		cl::NullRange,
		NULL,
		&event_kernel);

	//queue.enqueueReadBuffer(
	//	buffer,
	//	false,
	//	0,
	//	h.get_size_in_bytes(),
	//	h.get_ptr(),
	//	NULL,
	//	&event);

	// and wait
	event_kernel.wait();
	timer.stop();

	cout << "kernel: " << duration_in_ms(event_kernel) << " ms" << endl;
	//cout << "d2h:    " << duration_in_ms(event) << " ms" << endl;
	cout << "Timer: " << timer.delta() << " ms" << endl;
}



void simple_benchmark_all()
{
	// the buffer
	const int size = 1024 * 10;
	Extent2 ext(size, size);
	UnpinnedBuffer<float> h(ext);
	h.alloc();
	init2(h);

	cout << "Using test set of size " 
		<< ext.get_number_of_elems() << " elements [" 
		<< h.get_size_in_bytes()/1024.0 << " KB]" << endl;

	// the program 
	const string code = read_file(kernel_filename);
	cl::Program::Sources sources(1, { code.c_str(), code.length() + 1 });

	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	for (auto& platform : platforms)
	{
		cout << "Platform '" << platform.getInfo<CL_PLATFORM_NAME>() << "'" << endl;

		vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		if (devices.size() == 0)
		{
			cout << "  no devices" << endl;
		}
		for (auto& device : devices)
		{
			cout << "  device '" << device.getInfo<CL_DEVICE_NAME>() << "'" << endl;
			init2(h);
			try
			{
				simple_benchmark(h, platform, device, sources);
				if (verify2(h))
				{
					cout << "    ok" << endl;
				}
				else
				{
					cout << "    verification error" << endl;
				}
			}
			catch (cl::Error e)
			{
				cerr << "  ERROR " << e.what() << ", code=" << e.err() << endl;
			}
			cout << endl;
		}
	}


	Timer timer;
	timer.start();
	host_seq(h);
	timer.stop();
	cout << "Seq: " << timer.delta() << " ms" << endl;

	timer.start();
	host_par(h);
	timer.stop();
	cout << "Par: " << timer.delta() << " ms" << endl;

	// clean up
	h.free();
}