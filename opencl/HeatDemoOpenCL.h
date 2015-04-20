/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#define __CL_ENABLE_EXCEPTIONS
#include "cl_1_2.hpp"

#include "Defs.h"
#include "IDemo.h"
#include <vector>
#include <memory>
#include "UnpinnedBuffer.h"
#include "ConvertType.h"

class HeatDemoOpenCL
	: public virtual IDemo
{

public:

	HeatDemoOpenCL();

	virtual ~HeatDemoOpenCL()
	{
	}

	void init(const int width, const int height);

	void render(uchar4* d_image);

	void update();

	void synchronize();

	void cleanup();

	void shutdown();

	void save(std::string filename);

	void unset_opengl()
	{
		using_opengl = false;
	}

private:

	typedef std::shared_ptr<UnpinnedBuffer<float>> buf_t;

	std::vector<buf_t> raw_bufs;

	buf_t heat;

	int current_buf;

	Extent2 ext;

	ConvertType convertType = ConvertType::HSL;

	std::shared_ptr<UnpinnedBuffer<uchar4>> image;

	bool using_opengl = true;

	const std::string kernel_filename = "../opencl/cl/heat_single.cl";

	const int platform_id = 1;		// Anpassen!

	const int device_id = 1;		// Anpassen!

	cl::Program program;
	cl::Context context;
	cl::CommandQueue queue;

	std::vector<cl::Buffer> bufs;
	cl::Buffer heat_buf;

};


