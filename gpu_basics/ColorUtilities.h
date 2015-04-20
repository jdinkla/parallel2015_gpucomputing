/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#ifdef NVCC
#include <vector_functions.h>
#endif

__device__ __host__ inline
uchar4 gray(const float percentage)
{
	const unsigned char v = (unsigned char)(percentage * 255);
	return make_uchar4(v, v, v, 255);
}

// see http://axonflux.com/handy-rgb-to-hsl-and-rgb-to-hsv-color-model-c

__device__ __host__ inline
float hue_to_rgb(const float p, const float q, const float _t)
{
	float t = _t;
	if (t < 0.0f) t += 1.0f;
	if (t > 1.0f) t -= 1.0f;
	if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
	if (t < 1.0f / 2.0f) return q;
	if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
	return p;
}

// see http://axonflux.com/handy-rgb-to-hsl-and-rgb-to-hsv-color-model-c

__device__ __host__ inline
uchar4 hsl_to_rgb(const float h, const float s, const float l)
{
	float r, g, b;

	if (s == 0.0f)
	{
		r = g = b = l;		// achromatic
	}
	else
	{
		const float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
		const float p = 2 * l - q;
		r = hue_to_rgb(p, q, h + 1.0f / 3.0f);
		g = hue_to_rgb(p, q, h);
		b = hue_to_rgb(p, q, h - 1.0f / 3.0f);
	}
	return make_uchar4((unsigned char)(r * 255.0f), (unsigned char)(g * 255.0f), (unsigned char)(b * 255.0f), 1);
}
