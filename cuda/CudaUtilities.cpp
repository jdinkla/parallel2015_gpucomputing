/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "CudaUtilities.h"

// this has to be before the OP dependent stuf
#include "Defs.h"

// OpenGL, GLUT, GLEW, etc.
#ifdef WINDOWS
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#ifdef LINUX
#include <GL/freeglut.h>
#endif

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;

namespace CUDA
{

void check_rc(cudaError_t rc, const char* msg)
{
	if (rc != cudaSuccess)
	{
		throw std::runtime_error(msg);
	}
}

void check_cuda(const char* msg)
{
	check(msg);
}

void check(const char* msg)
{
	cudaError_t rc = cudaGetLastError();
	if (msg)
	{
		std::ostringstream buf;
		buf << msg << " - " << cudaGetErrorString(rc);
		check_rc(rc, buf.str().c_str());
	}
	else
	{
		check_rc(rc, cudaGetErrorString(rc));
	}
}

size_t get_free_device_mem()
{
	size_t free, total;
	cudaError_t rc = cudaMemGetInfo(&free, &total);
	check("get_free_device_mem");
	return free;
}

size_t get_total_device_mem()
{
	size_t free, total;
	cudaError_t rc = cudaMemGetInfo(&free, &total);
	check("get_total_device_mem");
	return total;
}

int get_number_of_gpus()
{
	int count = -1;
	cudaError_t rc = cudaGetDeviceCount(&count);
	check_rc(rc, "get_number_of_gpus");
	return count;
}

void reset_all()
{
	int count = -1;
	cudaError_t rc = cudaGetDeviceCount(&count);
	if (rc != cudaSuccess)
	{
		cerr << "ERROR: cudaGetDeviceCount" << endl;
		count = 1;
	}
	for (int d = 0; d < count; d++)
	{
		cudaSetDevice(d);
		cudaDeviceReset();
	}
}

size_t determine_max_mem(const int device_id, const size_t total)
{
	size_t upper = total;
	cudaSetDevice(device_id);
	CUDA::check("cudaSetDevice");

	size_t lower = 0;
	size_t max = 0;

	bool found = false;
	void* ptr;

	while (!found)
	{
		size_t middle = lower + (upper - lower) / 2;
		// try to allocate the memory
		cudaError_t rc = cudaMalloc(&ptr, middle);
		if (rc != cudaSuccess)
		{
			upper = middle;
			cudaFree(ptr);
			cudaGetLastError();
		}
		else
		{
			max = std::max(middle, max);
			cudaError_t rc2 = cudaFree(ptr);
			if (rc2 != cudaSuccess)
			{
				std::cerr << "Warning cudaFree has returned an error " << cudaGetErrorString(rc2) << endl;
			}
			lower = middle;
		}
		found = (upper - lower) < 128 * (1 << 20);
	}
	return max;
}

int get_opengl_device()
{
	unsigned int num_gl_devices = -1;
	const int count = 10;
	int cuda_devices[count];

	cudaGLGetDevices(&num_gl_devices, cuda_devices, count, cudaGLDeviceListCurrentFrame);
	CUDA::check("cudaGLGetDevices");

	return (num_gl_devices <= 0) ? -1 : cuda_devices[0];
}

void set_device_according_to_opengl()
{
	// set CUDA device to the device that is used by OpenGL
	int device = CUDA::get_opengl_device();
	if (device >= 0)
	{
		cudaSetDevice(device);
		CUDA::check("cudaSetDevice");
	}
	else
	{
		cerr << "Warning: can't set device" << endl;
	}
}

void sync_all()
{
	int count = -1;
	cudaError_t rc = cudaGetDeviceCount(&count);
	if (rc != cudaSuccess)
	{
		cerr << "ERROR: cudaGetDeviceCount" << endl;
		count = 1;
	}
	for (int d = 0; d < count; d++)
	{
		cudaSetDevice(d);
		cudaDeviceSynchronize();
	}
}


}
