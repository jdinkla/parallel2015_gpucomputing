/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include "ColorUtilities.h"

struct Convert_to_gray_functor
{
	const float max_val;

	Convert_to_gray_functor(const float _max_val)
		: max_val(_max_val)
	{
	}

	__device__ __host__
	uchar4 operator()(const float v) const
	{
		const float p = (v < max_val ? v : max_val) / max_val;
		return gray(p);
	}
};

struct Convert_hsl_functor
{
	const float max_val;

	Convert_hsl_functor(const float _max_val)
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

struct Convert_hsl2_functor
{
	const float max_val;

	Convert_hsl2_functor(const float _max_val)
		: max_val(_max_val)
	{
	}

	__device__ __host__
	uchar4 operator()(const float v) const
	{
		const float p = (v < max_val ? v : max_val) / max_val;
		return hsl_to_rgb(p, 1.0f, 0.25f + p / 2);
	}
};

// Generic converter
struct Convert_functor
{
	const float max_val;
	const int num_colors;
	uchar4* colors;

	Convert_functor(const float _max_val, const int _num_colors, uchar4* _colors)
		: max_val(_max_val)
		, num_colors(_num_colors)
		, colors(_colors)
	{
	}

	__device__ __host__
	uchar4 operator()(const float v) const
	{
		const float p = (v < max_val ? v : max_val) / max_val;
		const int idx = (int) (p * num_colors);
		return colors[idx];
	}

};


