/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "IHeatSource.h"
#include <memory>
#include <random>
#include <functional>

class RandomHeatSource
	: public virtual IHeatSource
{
public:

	void generate(float* ptr, const Extent2& ext, const int num_sources)
	{
		const size_t num_elems = ext.get_number_of_elems();
		// set all to 0.0f
		float* ptr_c = ptr;
		for (int i = 0; i < num_elems; i++)
		{
			*ptr_c++ = 0.0f;
		}

		// now generate random heat sources

		std::random_device rd;
		std::default_random_engine generator(rd());

		std::uniform_int_distribution<int> distribution_x(margin, ext.get_width() - margin);
		std::uniform_int_distribution<int> distribution_y(margin, ext.get_height() - margin);
		std::uniform_real_distribution<float> distribution_t(min_temp, max_temp);

#if defined(MAC) || defined(LINUX)
		for (int i = 0; i < num_sources; i++)
		{
			const int x = distribution_x(generator);
			const int y = distribution_y(generator);
			const float t = distribution_t(generator);
			std::cout << x << "," << y << ":" << t << std::endl;
			ptr[ext.index(x, y)] = t;
		}
#else
		auto dice_x = std::bind(distribution_x, generator);
		auto dice_y = std::bind(distribution_y, generator);
		auto dice_t = std::bind(distribution_t, generator);

		for (int i = 0; i < num_sources; i++)
		{
			ptr[ext.index(dice_x(), dice_y())] = dice_t();
		}
#endif
	}

	static std::shared_ptr<IHeatSource> make_shared()
	{
		return std::shared_ptr<IHeatSource>(new RandomHeatSource());
	}

private:

	const int margin = 20;

	const float min_temp = 0.0f;

	const float max_temp = 100.0f;

};