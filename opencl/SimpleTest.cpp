/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#define __CL_ENABLE_EXCEPTIONS

#include "SimpleTest.h"
#include "cl_1_2.hpp"
#include "UnpinnedBuffer.h"
#include <algorithm>
#include "OpenCLUtilities.h"
#include "FileUtilities.h"
#include <map>

static const char* kernel_filename = "../opencl/cl/square.cl";

using namespace std;

void simple_test(UnpinnedBuffer<float>& h, 
	cl::Platform& platform, 
	cl::Device& device,
	cl::Program::Sources& sources)
{
	cl::Context context   = get_context(platform, device);
	cl::CommandQueue queue(context, device, 0);

	// program

	cl::Program program = cl::Program(context, sources);
	program.build({device});
	cl::Kernel square_kernel(program, "square");

	// kernel & buffer
	cl::Buffer buffer(context, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
		h.get_size_in_bytes(), h.get_ptr());

	square_kernel.setArg(0, buffer);

	// fill the queue
	cl::Event event;
/*
	queue.enqueueWriteBuffer(
		buffer,
		false,
		0,
		h.get_size_in_bytes(),
		h.get_ptr(),
		NULL);
*/

	queue.enqueueNDRangeKernel(
	       square_kernel,
	       cl::NullRange,
	       cl::NDRange(h.get_extent().get_number_of_elems()),
	       cl::NullRange,
	       NULL);

	queue.enqueueReadBuffer(
		buffer,
		false,
		0,
		h.get_size_in_bytes(),
		h.get_ptr(),
		NULL,
		&event);

	// and wait
	event.wait();


}

// initializes the buffer with i*i
void init(UnpinnedBuffer<float>& h) 
{
	int i = 1;
	generate(h.begin(), h.end(), [&i]() { return i++; });
}

// verifies the buffer
bool verify(UnpinnedBuffer<float>& h)
{
	int i = 1;
	return std::all_of(h.begin(), h.end(), [&i](int v) { return v == i*i++; });
}

void simple_test_all()
{
	// the buffer
	const int size = 3;
	Extent2 ext(size, size);
	UnpinnedBuffer<float> h(ext);
	h.alloc();
	init(h);
	cout << "input" << endl;
	show(h);

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
			init(h);
			try
			{
				simple_test(h, platform, device, sources);
				if (verify(h))
				{
					cout << "    ok" << endl;
				}
				else
				{
					cout << "    error " << show_as_vector(h) << endl;
				}
			}
			catch (cl::Error e)
			{
				cerr << "  ERROR " << e.what() << ", code=" << e.err() << endl;
			}
			cout << endl;
		}
	}

	// clean up
	h.free();
}


// this is a test for my laptop
void simple_test_laptop()
{
	const int size = 3;
	Extent2 ext(size, size);
	UnpinnedBuffer<float> h(ext);
	h.alloc();
	init(h);
	cout << "input" << endl;
	show(h);

	// basics
	cl::Platform platform = get_platform_by_name("Apple");

	const string code = read_file(kernel_filename);
	cl::Program::Sources sources(1, {code.c_str(), code.length()+1});

	cl::Device device = get_device_by_name(platform, "GeForce GT 750M");
	simple_test(h, platform, device, sources);
	// show results
	cout << "results" << endl;
	show(h);

	cl::Device device2     = get_device_by_name(platform, "i7-4960HQ");
	init(h);
	simple_test(h, platform, device2, sources);
	// show results
	cout << "results" << endl;
	show(h);

	cl::Device device3     = get_device_by_name(platform, "Iris Pro");
	init(h);
	simple_test(h, platform, device2, sources);
	cout << "results" << endl;
	show(h);
}

void simple_test_multi_gpu()
{
	// the buffer
	//const int size = 10 * 1024;
	//auto show = [](UnpinnedBuffer<float>& x) {};
	const int size = 3;
	Extent2 ext(size, size);
	UnpinnedBuffer<float> h(ext);
	h.alloc();
	init(h);
	cout << "input" << endl;
	show(h);

	// tone buffer as output for each device
	vector<UnpinnedBuffer<float>*> host_bufs;

	// the sources 
	const string code = read_file(kernel_filename);
	cl::Program::Sources sources(1, { code.c_str(), code.length() + 1 });

	// platform
	cl::Platform platform = get_platform_by_name("NVIDIA");
	cout << "Platform '" << platform.getInfo<CL_PLATFORM_NAME>() << "'" << endl;

	// devices
	vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	const int numDevices = devices.size();
	if (numDevices == 0)
	{
		cout << "  no devices" << endl;
		return;
	}

	// context
	cl_context_properties properties[]
		= { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	cl::Context context(devices, properties);

	// command queues
	vector<cl::CommandQueue*> queues;
	for (auto& device : devices)
	{
		cout << "  device '" << device.getInfo<CL_DEVICE_NAME>() << "'" << endl;
		queues.push_back(new cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE));

		auto buf = new UnpinnedBuffer<float>(ext);
		buf->alloc();
		host_bufs.push_back(buf);
	}

	// buffer
	cl::Buffer buffer(context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		h.get_size_in_bytes(), h.get_ptr());

	// program and kernel
	cl::Program program = cl::Program(context, sources);
	program.build(devices);
	cl::Kernel square_kernel(program, "square");
	square_kernel.setArg(0, buffer);

	int d = 0;
	for (auto& device : devices)
	{

		queues[d]->enqueueWriteBuffer(
			buffer,
			false,
			0,
			h.get_size_in_bytes(),
			h.get_ptr(),
			NULL);

		queues[d]->enqueueNDRangeKernel(
			square_kernel,
			cl::NullRange,
			cl::NDRange(h.get_extent().get_number_of_elems()),
			cl::NullRange,
			NULL);

		queues[d]->enqueueReadBuffer(
			buffer,
			false,
			0,
			h.get_size_in_bytes(),
			host_bufs[d]->get_ptr(),
			NULL);

		d++;
	}

	// wait
	for (auto queue : queues)
	{
		queue->flush();
	}

	// Show results
	d = 0;
	for (auto& buf : host_bufs)
	{
		cout << "output: " << d << endl;
		show(*buf);
		d++;
	}

	// clean up
	for (auto& buf : host_bufs)
	{
		buf->free();
	}
	queues.clear();

}

