/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "IHeatSource.h"
#include <memory>

class ConstantHeatSource
	: public virtual IHeatSource
{

public:

	ConstantHeatSource(const int _step = 57, const int _mod = 100)
		: step(_step), mod(_mod)
	{
	}

	void generate(float* ptr, const Extent2& ext, const int num_sources)
	{
		const size_t num_elems = ext.get_number_of_elems();
		// set all to 0.0f
		float* ptr_c = ptr;
		for (int i = 0; i < num_elems; i++)
		{
			*ptr_c++ = 0.0f;
		}

		// now generate a constant pattern of sources
		for (int y = 0; y < ext.get_height(); y += step)
		{
			for (int x = 0; x < ext.get_width(); x += step)
			{
				ptr[ext.index(x, y)] = (float)(((x + 1) * (y + 1)) % mod);
			}
		}
	}

	static std::shared_ptr<IHeatSource> make_shared(const int step = 57, const int mod = 100)
	{
		return std::shared_ptr<IHeatSource>(new ConstantHeatSource(step, mod));
	}

private:

	const int step;

	const int mod;

};