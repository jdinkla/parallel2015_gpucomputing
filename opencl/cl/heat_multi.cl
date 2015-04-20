/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*
*
* kernels for the heat simulation - single device version
*
*/

__kernel void mask(__global float* mask, __global float* dest, int width, int height, int offset_y)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (x < width && y < height)
	{
		const int idx_src = y * width + x;
		const float value = mask[idx_src];
		if (value)
		{
			const int idx_dest = (y + offset_y) * width + x;
			dest[idx_dest] = value;
		}
	}
}

inline float get(__global float* src, int width, int height, int x, int y, float default_value)
{
	if (0 <= x && x < width && 0 <= y && y < height)
	{
		return src[y * width + x];
	}
	else
	{
		return default_value;
	}
}

__kernel void stencil(__global float* src, __global float* dest, 
	int width, int height_with, int height_without,
	const int offset_y, const float value)
{
	const int x = get_global_id(0);
	int y = get_global_id(1);
	if (x < width && y < height_without)
	{
		y += offset_y;
		const int idx = y * width + x;
		const float c = src[idx];
		const float l = get(src, width, height_with, x - 1, y, c);
		const float r = get(src, width, height_with, x + 1, y, c);
		const float t = get(src, width, height_with, x, y - 1, c);
		const float b = get(src, width, height_with, x, y + 1, c);
		dest[idx] = c + value * (t + b + l + r - 4 * c);;
	}
}


inline uchar4 gray(const float percentage)
{
	const unsigned char v = (unsigned char)(percentage * 255);
	return (uchar4)(v, v, v, 1);
}

__kernel void map_gray(__global float* src, __global uchar4* dest, int width, int height, float max_val)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (x < width && y < height)
	{
		const int idx = y * width + x;
		const float v = src[idx];
		const float p = (v < max_val ? v : max_val) / max_val;
		dest[idx] = gray(p);
	}
}

// see http://axonflux.com/handy-rgb-to-hsl-and-rgb-to-hsv-color-model-c

inline
float hue_to_rgb(const float p, const float q, const float _t)
{
	float t = _t;
	if (t < 0.0f) t += 1;
	if (t > 1.0f) t -= 1;
	if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
	if (t < 1.0f / 2.0f) return q;
	if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
	return p;
}

inline uchar4 hsl_to_rgb(const float h, const float s, const float l)
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
	return (uchar4)((unsigned char)(r * 255.0f), (unsigned char)(g * 255.0f), (unsigned char)(b * 255.0f), 1);
}

__kernel void map_hsl(__global float* src, __global uchar4* dest, int width, int height, float max_val)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (x < width && y < height)
	{
		const int idx = y * width + x;
		const float v = src[idx];
		const float p = (v < max_val ? v : max_val) / max_val;
		dest[idx] = hsl_to_rgb(p, 0.5f, 0.5f);
	}
}
