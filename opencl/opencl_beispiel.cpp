/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#define __CL_ENABLE_EXCEPTIONS
#include "cl_1_2.hpp"

#include <vector>
#include "FileUtilities.h"
#include <algorithm>
#include <iostream>
#include "OpenCLUtilities.h"

using namespace std;

int* make_vector(const int size)
{
	int* ptr = (int*) malloc(size * sizeof(int));
	for (int i = 0; i < size; i++)
	{
		ptr[i] = (i + 1);
	}
	return ptr;
}

void print(const int* h, const int size)
{
	cout << "h=(";
	string sep = "";
	for (int i = 0; i < size; i++)
	{
		cout << sep << h[i]; sep = ",";
	}
	cout << ")" << endl;
}

void opencl_beispiel2()
{
	const int num_elems = 16;
	int* malloced = make_vector(num_elems);
	const size_t sz = num_elems * sizeof(int);

	string code = read_file("../opencl/cl/square_int.cl");

	vector<cl::Platform> platforms; 	
	cl::Platform::get(&platforms);
	cl::Platform platform = platforms[1];

	vector<cl::Device> devices;			
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	cl::Device device = devices[0];

	cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	cl::Context context(device, properties);

	cl::Program::Sources sources(1, { code.c_str(), code.length() + 1 });
	cl::Program program = cl::Program(context, sources);
	program.build({ device });

	cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, malloced);

	cl::Kernel square_kernel(program, "square");
	square_kernel.setArg(0, buffer);

	cl::CommandQueue queue(context, device, 0);
	cl::Event event;

	queue.enqueueNDRangeKernel(square_kernel, cl::NullRange, cl::NDRange(16), cl::NullRange);
	queue.enqueueReadBuffer(buffer, true, 0, sz, malloced);

	print(malloced, num_elems);
	free(malloced);
}

void opencl_beispiel_orig()
{
	string code = read_file("../opencl/cl/square_int.cl");
	const int num_elems = 16; const size_t sz = num_elems * sizeof(int);
	int* malloced = make_vector(num_elems);
	vector<cl::Platform> platforms; cl::Platform::get(&platforms);
	vector<cl::Device> devices; platforms[1].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	cl_context_properties properties[]
		= { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[1])(), 0 };
	cl::Context context(devices[0], properties);
	cl::Program::Sources sources(1, { code.c_str(), code.length() + 1 });
	cl::Program program = cl::Program(context, sources); program.build({ devices[0] });
	cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, malloced);
	cl::Kernel square_kernel(program, "square"); square_kernel.setArg(0, buffer);
	cl::CommandQueue queue(context, devices[0], 0);
	queue.enqueueNDRangeKernel(square_kernel, cl::NullRange,
		cl::NDRange(16), cl::NullRange);
	queue.enqueueReadBuffer(buffer, true, 0, sz, malloced);
	print(malloced, num_elems); free(malloced);
}

void opencl_beispiel()
{
	string code = read_file("../opencl/cl/square_int.cl");
	const int num_elems = 16; const size_t sz = num_elems * sizeof(int);
	int* malloced = make_vector(num_elems);
	cl::Platform platform = get_platforms()[1];
	cl::Device device = get_devices(platform)[0];
	cl::Context context = get_context(platform, device);
	cl::Program program = get_program(device, context, code);
	cl::Buffer buffer(context, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, malloced);
	cl::Kernel square_kernel(program, "square"); 
	square_kernel.setArg(0, buffer);
	cl::CommandQueue queue(context, device, 0);
	queue.enqueueNDRangeKernel(square_kernel, cl::NullRange, 
		cl::NDRange(16), cl::NullRange);
	queue.enqueueReadBuffer(buffer, true, 0, sz, malloced);
	print(malloced, num_elems); free(malloced);
}

