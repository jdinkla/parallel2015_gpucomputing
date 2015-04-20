/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

namespace AMP
{

inline unsigned int make_uchar4(const int r, const int g, const int b, const int a) restrict(amp, cpu)
{
	return a << 24 | b << 16 | g << 8 | r;
}

inline unsigned int gray(const float percentage) restrict(amp, cpu)
{
	const int v = (int)(percentage * 255);
	return make_uchar4(v, v, v, 255);
}

// see http://axonflux.com/handy-rgb-to-hsl-and-rgb-to-hsv-color-model-c

inline float hue_to_rgb(const float p, const float q, const float _t) restrict(amp, cpu)
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

inline unsigned int hsl_to_rgb(const float h, const float s, const float l) restrict(amp, cpu)
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
	return make_uchar4((int)(r * 255.0f), (int)(g * 255.0f), (int)(b * 255.0f), 255);
}

}
