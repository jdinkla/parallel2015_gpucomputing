/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "HeatDemoCPU.h"
#include <iostream>
#include "Logger.h"
#include "MapHost.h"
#include "MaskHost.h"
#include "StencilHeatHost.h"
#include "HeatDemoDefs.h"
#include "Convert.h"
#include "FileUtilities.h"
#ifdef LINUX
#include <string.h>
#endif

using namespace std;

HeatDemoCPU::HeatDemoCPU()
{
}

void HeatDemoCPU::init_heat()
{
	heat = UnpinnedBuffer<float>::make_shared(ext);
	heat->alloc();
	fill(heat->begin(), heat->end(), 0.0f);				// set all to zero
	const int num_points = (int)(ext.get_number_of_elems() * ratio_of_points);
	get_heat_source()->generate(heat->get_ptr(), ext, num_points);
}

void HeatDemoCPU::init(const int _width, const int _height)
{
	DEBUG("init");
	ext = Extent2(_width, _height);
	init_heat();

	image = UnpinnedBuffer<uchar4>::make_shared(ext);
	image->alloc();

	buf_t d0 = UnpinnedBuffer<float>::make_shared(ext);
	d0->alloc();
	bufs.push_back(d0);

	buf_t d1 = UnpinnedBuffer<float>::make_shared(ext);
	d1->alloc();
	bufs.push_back(d1);

	fill(d0->begin(), d0->end(), 0.0f);					// fill d0

	// we calculated buffer 0, so the next 'one' is 1
	current_buf = 1;
}

void HeatDemoCPU::render(uchar4* d_image)
{
	DEBUG("render");

	if (!using_opengl)
	{
		return;
	}

	//Convert_to_gray_functor f(max_heat);
	Convert_hsl_functor f(max_heat);

	if (is_parallel)
	{
		map_2d_par<float, uchar4>(
			bufs[!current_buf]->get_ptr(),
			image->get_ptr(),
			ext,
			[&f](const float value) { return f(value); }
		);
	}
	else
	{
		map_2d<float, uchar4>(
			bufs[!current_buf]->get_ptr(),
			image->get_ptr(),
			ext,
			[&f](const float value) { return f(value); }
		);
	}

	// if d_image is on the device, we need an external copy function
	if (copy_fun)
	{
		copy_fun(d_image, image);
	}
	else
	{
		::memcpy(d_image, image->get_ptr(), image->get_size_in_bytes());
	}
}

void HeatDemoCPU::update()
{
	DEBUG("update");

	UnpinnedBuffer<float>& previous = *bufs[!current_buf].get();
	UnpinnedBuffer<float>& current = *bufs[current_buf].get();
	UnpinnedBuffer<float>& heat_buf = *heat.get();

	if (is_parallel)
	{
		// add_heat();
		mask_2d_par<float>(heat_buf, previous);

		// stencil();
		stencil_heat_2d_par(previous, current, ct);
	}
	else
	{
		// add_heat();
		mask_2d<float>(heat_buf, previous);

		// stencil();
		stencil_heat_2d(previous, current, ct);
	}

	current_buf = !current_buf;
}

void HeatDemoCPU::synchronize()
{
}

void HeatDemoCPU::cleanup()
{
	DEBUG("cleanup");
	heat->free();
	bufs[0]->free();
	bufs[1]->free();
	bufs.clear();
	image->free();
}

void HeatDemoCPU::shutdown()
{
}

void HeatDemoCPU::save(std::string filename)
{
	if (!using_opengl)
	{
		Convert_hsl_functor f(max_heat);
		map_2d<float, uchar4>(
			bufs[!current_buf]->get_ptr(),
			image->get_ptr(),
			ext,
			[&f](const float value) { return f(value); }
		);
	}
	save_bmp(filename, image->get_ptr(), ext.get_width(), ext.get_height());
}
