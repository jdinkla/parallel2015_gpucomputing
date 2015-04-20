/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#define __device__
#define __host__

typedef unsigned char uchar;
//typedef struct { uchar x, y, z, w; } uchar4;

struct __declspec(align(4)) uchar4
{
	unsigned char x, y, z, w;
};

inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
	uchar4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

// dim3
