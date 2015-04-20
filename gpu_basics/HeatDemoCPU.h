/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include "IDemo.h"
#include <vector>
#include <memory>
#include "UnpinnedBuffer.h"
#include "ConvertType.h"

class HeatDemoCPU
	: public virtual IDemo
{

public:

	HeatDemoCPU();

	virtual ~HeatDemoCPU()
	{
	}

	void init(const int width, const int height);

	void render(uchar4* d_image);

	void update();

	void synchronize();

	void cleanup();

	void shutdown();

	void save(std::string filename);

	void unset_opengl()
	{
		using_opengl = false;
	}

	void set_parallel(const bool _parallel)
	{
		is_parallel = _parallel;
	}

	using buf_t = std::shared_ptr<UnpinnedBuffer<float>>;
	using image_t = std::shared_ptr<UnpinnedBuffer<uchar4>>;

	std::function<void(uchar4*, image_t)> copy_fun = nullptr;

private:

	void init_heat();

	std::vector<buf_t> bufs;

	buf_t heat;

	int current_buf;

	Extent2 ext;

	const float ratio_of_points = 0.0001f;

	ConvertType convertType = ConvertType::HSL;

	image_t image;

	// should more than one thread be used?
	bool is_parallel = true;

	bool using_opengl = true;
};


