/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#define _CRT_SECURE_NO_WARNINGS

#include "FileUtilities.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "UnpinnedBuffer.h"
#include "Convert.h"
#include "MapHost.h"

using namespace std;

string read_file(string name) 
{
    std::ifstream t(name);
    if (t.fail()) 
    {
    	throw runtime_error("file does not exist");
    }
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

void save_bmp(string filename, uchar4* ptr, const Extent2& ext)
{
	save_bmp(filename, ptr, ext.get_width(), ext.get_height());
}

// improved and adapted version of http://stackoverflow.com/a/2654860
void save_bmp(string filename, uchar4* ptr, const int width, const int height)
{
	const int num_elems = width * height;
	unsigned char* img = (unsigned char*)malloc(3 * num_elems);

	int i = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			const uchar4 value = ptr[i++];
			const int j = (height - y - 1) * width + x;
			img[3 * j + 2] = value.x;
			img[3 * j + 1] = value.y;
			img[3 * j + 0] = value.z;
		}
	}

	unsigned char bmpfileheader[14] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
	unsigned char bmpinfoheader[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0 };
	unsigned char bmppad[3] = { 0, 0, 0 };

	const int filesize = 54 + 3 * num_elems;
	bmpfileheader[2] = (unsigned char)(filesize);
	bmpfileheader[3] = (unsigned char)(filesize >> 8);
	bmpfileheader[4] = (unsigned char)(filesize >> 16);
	bmpfileheader[5] = (unsigned char)(filesize >> 24);

	bmpinfoheader[4] = (unsigned char)(width);
	bmpinfoheader[5] = (unsigned char)(width >> 8);
	bmpinfoheader[6] = (unsigned char)(width >> 16);
	bmpinfoheader[7] = (unsigned char)(width >> 24);
	bmpinfoheader[8] = (unsigned char)(height);
	bmpinfoheader[9] = (unsigned char)(height >> 8);
	bmpinfoheader[10] = (unsigned char)(height >> 16);
	bmpinfoheader[11] = (unsigned char)(height >> 24);

	FILE *f = fopen(filename.c_str(), "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);
	for (int i = 0; i < height; i++)
	{
		fwrite(img + (width*(height - i - 1) * 3), 3, width, f);
		fwrite(bmppad, 1, (4 - (width * 3) % 4) % 4, f);
	}
	fclose(f);

}

void save_bmp(string filename, HostBuffer<float>& host, const float max_heat)
{
	UnpinnedBuffer<uchar4> image(host.get_extent());
	image.alloc();
	Convert_hsl_functor f(max_heat);

#if defined(MAC) || defined(LINUX)
	// inline, because clang doesn't find the template function
	const Extent2& ext = host.get_extent();
	const int h = ext.get_height();
	AsyncSchedulerInterval<int> s;
	float* src = host.get_ptr();
	uchar4* dest = image.get_ptr();
	s.sync(0, h, [src, dest, ext, f](const int y)
	{
		// map_2d_row(src, dest, ext, f, y);
		const int w = ext.get_width();
		int idx = ext.index(0, y);
		for (int x = 0; x < w; x++)
		{
			dest[idx] = f(src[idx]);
			idx++;
		}
	});
#else
	map_2d_par<float, uchar4>(
		host.get_ptr(),
		image.get_ptr(),
		host.get_extent(),
		[&f](const float value) { return f(value); }
	);
#endif
	save_bmp(filename, image.get_ptr(), host.get_extent());
}