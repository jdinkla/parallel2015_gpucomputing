/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "HeatDemoAMP.h"
#include <iostream>
#include "Logger.h"
#include "HeatDemoDefs.h"
#include "Extent2AMP.h"
#include "ColorUtilitiesAMP.h"
#include "FileUtilities.h"
#include "AMPUtilities.h"

using namespace std;

using concurrency::index;
using concurrency::parallel_for_each;
using concurrency::array_view;
using namespace concurrency::diagnostic;
using concurrency::accelerator;

HeatDemoAMP::HeatDemoAMP()
{
	accelerator::set_default(get_accs()[accelerator_to_use].device_path);
}

void HeatDemoAMP::init(const int _width, const int _height)
{
	DEBUG("init"); span mysp(mySeries, _T("init"));

	ext = Extent2(_width, _height);

	// heat
	heat_buf = UnpinnedBuffer<float>::make_shared(ext);
	heat_buf->alloc();
	const int num_points = (int)(ext.get_number_of_elems() * ratio_of_heat_sources);
	get_heat_source()->generate(heat_buf->get_ptr(), ext, num_points);

	heat_view = new array_view<const float, 2>(ext.get_height(), ext.get_width(), heat_buf->get_ptr());

	image_buf = UnpinnedBuffer<uchar4>::make_shared(ext);
	image_buf->alloc();

	image_view = new array_view<unsigned int, 2>(ext.get_height(), ext.get_width(), (unsigned int*)image_buf->get_ptr());

	buf_t d0 = UnpinnedBuffer<float>::make_shared(ext);
	d0->alloc();

	buf_t d1 = UnpinnedBuffer<float>::make_shared(ext);
	d1->alloc();

	fill(d0->begin(), d0->end(), 0.0f);					// fill d0

	view_t* d0_view = new array_view<float, 2>(ext.get_height(), ext.get_width(), d0->get_ptr());
	view_t* d1_view = new array_view<float, 2>(ext.get_height(), ext.get_width(), d1->get_ptr());

	bufs = new DoubleBuffer<buf_t>(d0, d1);
	views = new DoubleBuffer<view_t*>(d0_view, d1_view);

	// we calculated buffer 0, so the next 'one' is 1
	views->swap();
}

void HeatDemoAMP::render(uchar4* d_image)
{
	//if (!using_opengl)
	//{
	//	return;
	//}

	DEBUG("render"); span mysp(mySeries, _T("render"));

	view_t& previous = *(views->get_previous());
	view_t& current = *(views->get_current());

	const float max_val = max_heat;

	previous.discard_data();
	image_view->discard_data();
	
	array_view<unsigned int, 2>& image_v = *image_view;			// pointer are not allowed in restrict(amp)

	parallel_for_each(image_v.extent, [=](index<2> idx) restrict(amp)
	{
		const float v = current(idx);
		const float p = (v < max_val ? v : max_val) / max_val;
		//image_v(idx) = gray(p);
		image_v(idx) = AMP::hsl_to_rgb(p, 0.5f, 0.5f);
	});

	// get data to host
	image_view->synchronize();

	// copy
	memcpy(d_image, image_buf->get_ptr(), image_buf->get_size_in_bytes());
}

void HeatDemoAMP::update()
{
	DEBUG("update"); span mysp(mySeries, _T("update"));

	view_t& previous = *(views->get_previous());
	view_t& current = *(views->get_current());

	// add_heat();
	if (is_add_heat())
	{
		// mask_2d<float>(heat_buf, previous);
		current.discard_data();
		image_view->discard_data();

		array_view<const float, 2>& heat_v = *heat_view;			// pointer are not allowed in restrict(amp)

		parallel_for_each(previous.extent, [=](index<2> idx) restrict(amp)
		{
			const float value = heat_v(idx);
			if (value)
			{
				previous(idx) = value;
			}
		});
	}

	// stencil();
	// stencil_heat_2d_par(previous, current, ct);
	float ct_val = ct;
	const Extent2AMP e(ext);

	current.discard_data();
	previous.discard_data();
	image_view->discard_data();

	parallel_for_each(current.extent, [=](index<2> idx) restrict(amp)
	{
		auto get = [=](const int x, const int y, const float default_value) restrict(amp)
		{
			index<2> jdx{ y, x };
			return e.in_bounds_strict(x, y) ? previous(jdx) : default_value;
		};
		const int x = idx[1];
		const int y = idx[0];
		const float c = previous(idx);
		const float l = get(x - 1, y, c);
		const float r = get(x + 1, y, c);
		const float t = get(x, y - 1, c);
		const float b = get(x, y + 1, c);
		current(idx) = c + ct_val * (t + b + l + r - 4 * c);
	});

	views->swap();
}

void HeatDemoAMP::synchronize()
{
	view_t& previous = *(views->get_previous());
	view_t& current = *(views->get_current());

	previous.synchronize();
	current.synchronize();
}

void HeatDemoAMP::cleanup()
{
	DEBUG("cleanup"); span mysp(mySeries, _T("cleanup"));
	delete heat_view;
	delete image_view;
	heat_buf->free();
	bufs->get_current()->free();
	bufs->get_previous()->free();
	image_buf->free();
}

void HeatDemoAMP::save(std::string filename)
{
	UnpinnedBuffer<uchar4> image(ext);
	image.alloc();

	view_t& previous = *(views->get_previous());
	view_t& current = *(views->get_current());

	const float max_val = max_heat;

	previous.discard_data();
	image_view->discard_data();

	array_view<unsigned int, 2>& image_v = *image_view;			// pointer are not allowed in restrict(amp)

	parallel_for_each(image_v.extent, [=](index<2> idx) restrict(amp)
	{
		const float v = current(idx);
		const float p = (v < max_val ? v : max_val) / max_val;
		image_v(idx) = AMP::hsl_to_rgb(p, 0.5f, 0.5f);
	});

	// get data to host
	image_view->synchronize();

	// copy
	memcpy(image.get_ptr(), image_buf->get_ptr(), image_buf->get_size_in_bytes());

	save_bmp(filename, image.get_ptr(), ext.get_width(), ext.get_height());
}

void HeatDemoAMP::shutdown()
{
	DEBUG("shutdown"); span mysp(mySeries, _T("shutdown"));
}
