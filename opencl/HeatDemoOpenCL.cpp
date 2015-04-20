/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "HeatDemoOpenCL.h"
#include <iostream>
#include "Logger.h"
#include "MapHost.h"
#include "MaskHost.h"
#include "StencilHeatHost.h"
#include "HeatDemoDefs.h"
#include "Convert.h"
#include "OpenCLUtilities.h"
#include "FileUtilities.h"

using namespace std;

HeatDemoOpenCL::HeatDemoOpenCL()
{
	cl::Platform platform = get_platforms()[platform_id];
	cl::Device device = get_devices(platform)[device_id];
	context = get_context(platform, device);
	queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
		
	const string code = read_file(kernel_filename);
	cl::Program::Sources sources(1, { code.c_str(), code.length() + 1 });
	program = cl::Program(context, sources); 
	program.build({ device });
}

void HeatDemoOpenCL::init(const int _width, const int _height)
{
	DEBUG("init");
	ext = Extent2(_width, _height);
	// init_heat();
	heat = make_shared<UnpinnedBuffer<float>>(ext);
	heat->alloc();
	const int num_points = (int)(ext.get_number_of_elems() * ratio_of_heat_sources);
	get_heat_source()->generate(heat->get_ptr(), ext, num_points);

	heat_buf = cl::Buffer(context,
		CL_MEM_READ_ONLY, // | CL_MEM_COPY_HOST_PTR,
		heat->get_size_in_bytes(), heat->get_ptr());

	queue.enqueueWriteBuffer(
		heat_buf,
		false,
		0,
		heat->get_size_in_bytes(),
		heat->get_ptr(),
		NULL);

	image = make_shared<UnpinnedBuffer<uchar4>>(ext);
	image->alloc();

	buf_t d0 = make_shared<UnpinnedBuffer<float>>(ext);
	d0->alloc();
	raw_bufs.push_back(d0);

	buf_t d1 = make_shared<UnpinnedBuffer<float>>(ext);
	d1->alloc();
	raw_bufs.push_back(d1);

	fill(d0->begin(), d0->end(), 0.0f);					// fill d0

	cl::Buffer buf_d0(context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		heat->get_size_in_bytes(), d0->get_ptr());
	bufs.push_back(buf_d0);

	cl::Buffer buf_d1(context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		heat->get_size_in_bytes(), d1->get_ptr());
	bufs.push_back(buf_d1);

	//queue.enqueueWriteBuffer(
	//	buf_d0,
	//	false,
	//	0,
	//	d0->get_size_in_bytes(),
	//	d0->get_ptr(),
	//	NULL);

	cl::Kernel zero_kernel(program, "zero");
	zero_kernel.setArg(0, bufs[0]);
	zero_kernel.setArg(1, ext.get_width());
	zero_kernel.setArg(2, ext.get_height());

	queue.enqueueNDRangeKernel(
		zero_kernel,
		cl::NullRange,
		cl::NDRange(ext.get_width(), ext.get_height()),
		cl::NullRange,
		NULL);

	// we calculated buffer 0, so the next 'one' is 1
	current_buf = 1;
}

void HeatDemoOpenCL::render(uchar4* d_image)
{
	DEBUG("render");

	if (!using_opengl)
	{
		return;
	}

	try
	{
		cl::Buffer image_buf(context, CL_MEM_WRITE_ONLY, heat->get_size_in_bytes(), d_image);
		cl::Kernel map_kernel(program, "map_hsl");
		map_kernel.setArg(0, bufs[!current_buf]);
		map_kernel.setArg(1, image_buf);
		map_kernel.setArg(2, ext.get_width());
		map_kernel.setArg(3, ext.get_height());
		map_kernel.setArg(4, max_heat);

		queue.enqueueNDRangeKernel(
			map_kernel,
			cl::NullRange,
			cl::NDRange(ext.get_width(), ext.get_height()),
			cl::NullRange,
			NULL);

		queue.enqueueReadBuffer(
			image_buf,
			false,
			0,
			heat->get_size_in_bytes(),
			d_image,
			NULL);

		queue.flush();
	}
	catch (cl::Error e)
	{
		cerr << "  ERROR " << e.what() << ", code=" << e.err() << endl;
	}

}

void HeatDemoOpenCL::update()
{
	DEBUG("update");

	cl::Buffer& previous = bufs[!current_buf];
	cl::Buffer& current = bufs[current_buf];

	try
	{
		// add_heat();
		if (is_add_heat())
		{
			cl::Kernel mask_kernel(program, "mask");
			mask_kernel.setArg(0, heat_buf);
			mask_kernel.setArg(1, previous);
			mask_kernel.setArg(2, ext.get_width());
			mask_kernel.setArg(3, ext.get_height());

			queue.enqueueNDRangeKernel(
				mask_kernel,
				cl::NullRange,
				cl::NDRange(ext.get_width(), ext.get_height()),
				cl::NullRange,
				NULL);
		}

		// stencil();
		cl::Kernel stencil_kernel(program, "stencil");
		stencil_kernel.setArg(0, previous);
		stencil_kernel.setArg(1, current);
		stencil_kernel.setArg(2, ext.get_width());
		stencil_kernel.setArg(3, ext.get_height());
		stencil_kernel.setArg(4, ct);

		queue.enqueueNDRangeKernel(
			stencil_kernel,
			cl::NullRange,
			cl::NDRange(ext.get_width(), ext.get_height()),
			cl::NullRange,
			NULL);

		//queue.flush();
	}
	catch (cl::Error e)
	{
		cerr << "  ERROR " << e.what() << ", code=" << e.err() << endl;
	}

	current_buf = !current_buf;
}

void HeatDemoOpenCL::cleanup()
{
	DEBUG("cleanup");

	bufs.clear();

	//cl::Buffer.release();

	heat->free();
	raw_bufs[0]->free();
	raw_bufs[1]->free();
	raw_bufs.clear();
	image->free();
}

void HeatDemoOpenCL::synchronize()
{
	queue.finish();
}

void HeatDemoOpenCL::save(std::string filename)
{
	try
	{
		UnpinnedBuffer<uchar4> image(ext);
		image.alloc();

		cl::Buffer image_buf(context, CL_MEM_WRITE_ONLY, image.get_size_in_bytes(), image.get_ptr());
		cl::Kernel map_kernel(program, "map_hsl");
		map_kernel.setArg(0, bufs[!current_buf]);
		map_kernel.setArg(1, image_buf);
		map_kernel.setArg(2, ext.get_width());
		map_kernel.setArg(3, ext.get_height());
		map_kernel.setArg(4, max_heat);

		queue.enqueueNDRangeKernel(
			map_kernel,
			cl::NullRange,
			cl::NDRange(ext.get_width(), ext.get_height()),
			cl::NullRange,
			NULL);

		queue.enqueueReadBuffer(
			image_buf,
			false,
			0,
			image.get_size_in_bytes(),
			image.get_ptr(),
			NULL);

		queue.flush();

		save_bmp(filename, image.get_ptr(), ext.get_width(), ext.get_height());
	}
	catch (cl::Error e)
	{
		cerr << "  ERROR " << e.what() << ", code=" << e.err() << endl;
	}
}

void HeatDemoOpenCL::shutdown()
{
}
