/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include "ColorUtilities.h"
#include <memory>

struct Converter
{
	virtual void init()
	{

	}

	virtual void shutdown()
	{
	}

	__device__ __host__
	virtual uchar4 operator()(const float v) const
	{
		return make_uchar4(0, 0, 0, 0);
	}

};

struct GrayConverter
	: public Converter
{
	const float max_val;

	GrayConverter(const float _max_val)
		: max_val(_max_val)
	{
	}

	__device__ __host__
	virtual uchar4 operator()(const float v) const override
	{
		const float p = (v < max_val ? v : max_val) / max_val;
		return gray(p);
	}
};

struct HSLConverter
	: public Converter
{
	const float max_val;

	HSLConverter(const float _max_val)
		: max_val(_max_val)
	{
	}

	__device__ __host__
	virtual uchar4 operator()(const float v) const override
	{
		const float p = (v < max_val ? v : max_val) / max_val;
		return hsl_to_rgb(p, 0.5f, 0.5f);
	}

};

struct HSLConverterX
{
	const float max_val;

	HSLConverterX(const float _max_val)
		: max_val(_max_val)
	{
	}

	__device__ __host__
	uchar4 operator()(const float v) const
	{
		const float p = (v < max_val ? v : max_val) / max_val;
		return hsl_to_rgb(p, 0.5f, 0.5f);
	}

};


struct HSLConverter2
	: public Converter
{
	const float max_val;
	const float low_hue;
	//const float high_hue;
	const float diff_hue;

	HSLConverter2(const float _max_val, const float _low_hue, const float _high_hue)
		: max_val(_max_val)
		, low_hue(_low_hue)
		//, high_hue(_high_hue)
		, diff_hue(_high_hue - low_hue)
	{
	}

	__device__ __host__
	inline float map(const float v) const
	{
		// v is in [0, 1], m is in [low, hi]
		return low_hue + v * diff_hue;
		//return v;
	}

	__device__ __host__
	virtual uchar4 operator()(const float v) const override
	{
		const float p = (v < max_val ? v : max_val) / max_val;
		//return hsl_to_rgb(map(p), 0.5f, 0.5f);
		return hsl_to_rgb(map(p), 0.5f, 0.5f);
	}

};

struct LookupConverter
	: public Converter
{
	const float max_val;
	const int num_colors;
	uchar4* colors;

	LookupConverter(const float _max_val, const int _num_colors, uchar4* _colors)
		: max_val(_max_val)
		, num_colors(_num_colors)
		, colors(_colors)
	{
	}

	virtual void init() override
	{
		//colors = std::shared_ptr<ManagedBuffer<uchar4>>(get_color_buffer());
	}

	virtual void shutdown() override
	{
		//if (colors)
		//{
		//	colors->free();
		//}
	}

	__device__ __host__
	virtual uchar4 operator()(const float v) const override
	{
		const float p = (v < max_val ? v : max_val) / max_val;
		const int idx = (int)(p * num_colors);
		return colors[idx];
	}
};

