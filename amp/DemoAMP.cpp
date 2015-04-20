/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "DemoAMP.h"
#include "UnpinnedBuffer.h"
#include <vector>
#include <amp.h>
#include "HeatDemoDefs.h"
#include "DoubleBuffer.h"
#include "Extent2.h"
#include "Extent2AMP.h"
#include "ColorUtilitiesAMP.h"
#include "FileUtilities.h"
#include "ConstantHeatSource.h"
#include "AMPUtilities.h"

using namespace std;
using concurrency::parallel_for_each;
using concurrency::array_view;
using concurrency::index;
using concurrency::accelerator;

const int w = small_width;
const int h = small_height;

static Extent2 ext(w, h);

typedef UnpinnedBuffer<float> buf_t;

inline buf_t* create_alloced()
{
	buf_t* d = new UnpinnedBuffer<float>(ext); 
	d->alloc();
	return d;
}

inline void init_heat(buf_t &heat)
{
	heat.alloc();									// buffer heat
	ConstantHeatSource c;
	c.generate(heat.get_ptr(), ext, num_heat_sources);
}

void demo_single_amp()
{
	float ct_val = ct;
	typedef UnpinnedBuffer<float> buf_t;

	buf_t heat(ext); init_heat(heat);
	buf_t* d0 = create_alloced(); buf_t* d1 = create_alloced();		// buffer data
	fill(d0->begin(), d0->end(), 0.0f);

	typedef array_view < float, 2 > av_t;							// views
	av_t av_heat(h, w, heat.get_ptr());								// heat sources
	av_t* av_0 = new av_t(h, w, d0->get_ptr());	
	av_t* av_1 = new av_t(h, w, d1->get_ptr());	

	DoubleBuffer<av_t*> avs{ av_0, av_1 };
	avs.swap();

	for (int i = 0; i < iterations; i++)
	{
		av_t& previous = *(avs.get_previous());
		av_t& current = *(avs.get_current());

		// add_heat();
		current.discard_data();
		parallel_for_each(previous.extent, [=](index<2> idx) restrict(amp)
		{
			const float value = av_heat[idx];
			if (value)
			{
				previous[idx] = value;
			}
		});
		previous.synchronize();
	
		// stencil
		const Extent2AMP e(ext);
		current.discard_data();
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
		current.synchronize();

		avs.swap();
	}

	//do_something_with_the_data(host);	

	// save the result
	const float max_val = max_heat;
	av_t& current = *(avs.get_previous());

	UnpinnedBuffer<uchar4> image_buf(ext); 	image_buf.alloc();
	array_view<unsigned int, 2> image_v(h, w, (unsigned int*)image_buf.get_ptr());

	//previous.discard_data();
	image_v.discard_data();
	parallel_for_each(image_v.extent, [=](index<2> idx) restrict(amp)
	{
		const float v = current(idx);
		const float p = (v < max_val ? v : max_val) / max_val;
		image_v(idx) = AMP::hsl_to_rgb(p, 0.5f, 0.5f);
	});

	// get data to host
	image_v.synchronize();

	// copy
	save_bmp("AMP single standalone.bmp", image_buf.get_ptr(), w, h);

	// Clean up
	av_0->discard_data();
	av_1->discard_data();
	av_heat.discard_data();

	heat.free();
	//host.free();
	d0->free();
	d1->free();

	// reset?
}

void demo_single_amp(concurrency::accelerator& acc, std::string filename)
{
	concurrency::accelerator_view view = acc.default_view;

	float ct_val = ct;
	typedef UnpinnedBuffer<float> buf_t;

	buf_t heat(ext); init_heat(heat);
	buf_t* d0 = create_alloced(); buf_t* d1 = create_alloced();		// buffer data
	fill(d0->begin(), d0->end(), 0.0f);

	typedef array_view < float, 2 > av_t;							// views
	av_t av_heat(h, w, heat.get_ptr());								// heat sources
	av_t* av_0 = new av_t(h, w, d0->get_ptr());
	av_t* av_1 = new av_t(h, w, d1->get_ptr());

	DoubleBuffer<av_t*> avs{ av_0, av_1 };
	avs.swap();

	for (int i = 0; i < iterations; i++)
	{
		av_t& previous = *(avs.get_previous());
		av_t& current = *(avs.get_current());

		// add_heat();
		current.discard_data();
		parallel_for_each(view, previous.extent, [=](index<2> idx) restrict(amp)
		{
			const float value = av_heat[idx];
			if (value)
			{
				previous[idx] = value;
			}
		});
		previous.synchronize();

		// stencil
		const Extent2AMP e(ext);
		current.discard_data();
		parallel_for_each(view, current.extent, [=](index<2> idx) restrict(amp)
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
		current.synchronize();

		avs.swap();
	}

	//do_something_with_the_data(host);	

	// save the result
	const float max_val = max_heat;
	av_t& current = *(avs.get_previous());

	UnpinnedBuffer<uchar4> image_buf(ext); 	image_buf.alloc();
	array_view<unsigned int, 2> image_v(h, w, (unsigned int*)image_buf.get_ptr());

	//previous.discard_data();
	image_v.discard_data();
	parallel_for_each(view, image_v.extent, [=](index<2> idx) restrict(amp)
	{
		const float v = current(idx);
		const float p = (v < max_val ? v : max_val) / max_val;
		image_v(idx) = AMP::hsl_to_rgb(p, 0.5f, 0.5f);
	});

	// get data to host
	image_v.synchronize();

	// copy
	save_bmp(filename, image_buf.get_ptr(), w, h);

	// Clean up
	av_0->discard_data();
	av_1->discard_data();
	av_heat.discard_data();

	heat.free();
	//host.free();
	d0->free();
	d1->free();

	// reset?
}

void demo_single_amp_all_devices()
{
	vector<accelerator> accs = get_accs();
	int i = 0;
	for (auto& acc : accs)
	{
		stringstream str;
		str << "AMP single standalone dev " << i++ << ".bmp";
		demo_single_amp(acc, str.str());
	}
}

void heat_demo_single_amp_reformatted()
{
	float ct_val = ct;

	typedef UnpinnedBuffer<float> buf_t;
	buf_t heat(ext); init_heat(heat);
	buf_t* d0 = create_alloced(); buf_t* d1 = create_alloced();		// buffer data
	fill(d0->begin(), d0->end(), 0.0f);

	typedef array_view < float, 2 > av_t;							// views
	av_t av_heat(h, w, heat.get_ptr());								// heat sources
	av_t* av0 = new av_t(h, w, d0->get_ptr());
	av_t* av1 = new av_t(h, w, d1->get_ptr());
	DoubleBuffer<av_t*> avs{ av0, av1 };
	avs.swap();

	for (int i = 0; i < iterations; i++) {
		av_t& prev = *(avs.get_previous()); av_t& curr = *(avs.get_current());
		// add_heat();
		curr.discard_data();
		parallel_for_each(prev.extent, [=](index<2> idx) restrict(amp)
		{
			const float value = av_heat[idx];
			if (value)
			{
				prev[idx] = value;
			}
		});
		prev.synchronize();
		// stencil()
		const Extent2AMP e(ext);
		curr.discard_data();
		parallel_for_each(curr.extent, [=](index<2> idx) restrict(amp)
		{
			auto get = [=](const int x, const int y, const float default_value) restrict(amp)
			{
				index<2> jdx{ y, x };
				return e.in_bounds_strict(x, y) ? prev(jdx) : default_value;
			};
			const int x = idx[1]; const int y = idx[0];
			const float c = prev(idx);
			const float l = get(x - 1, y, c);
			const float r = get(x + 1, y, c);
			const float t = get(x, y - 1, c);
			const float b = get(x, y + 1, c);
			curr(idx) = c + ct_val * (t + b + l + r - 4 * c);
		});
		curr.synchronize();

		avs.swap();
	}

	//do_something_with_the_data(host);	

	// save the result
	const float max_val = max_heat;
	av_t& current = *(avs.get_previous());

	UnpinnedBuffer<uchar4> image_buf(ext); 	image_buf.alloc();
	array_view<unsigned int, 2> image_v(h, w, (unsigned int*)image_buf.get_ptr());

	//previous.discard_data();
	image_v.discard_data();
	parallel_for_each(image_v.extent, [=](index<2> idx) restrict(amp)
	{
		const float v = current(idx);
		const float p = (v < max_val ? v : max_val) / max_val;
		image_v(idx) = AMP::hsl_to_rgb(p, 0.5f, 0.5f);
	});

	// get data to host
	image_v.synchronize();

	// copy
	save_bmp("AMP single standalone.bmp", image_buf.get_ptr(), w, h);

	// Clean up
	av0->discard_data();
	av1->discard_data();
	av_heat.discard_data();

	heat.free();
	//host.free();
	d0->free();
	d1->free();

	// reset?
}
