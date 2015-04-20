/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "IDemo.h"
#include <vector>
#include <memory>
#include "DeviceBuffer.h"
#include "Converter.h"
#include "HeatDemoDefs.h"
#include "DoubleBuffer.h"

class HeatDemoSingleGPU
	: public virtual IDemo
{

public:

	HeatDemoSingleGPU();

	virtual ~HeatDemoSingleGPU()
	{
	}

	void init(const int width, const int height);

	void render(uchar4* d_image);

	void update();

	void synchronize();

	void cleanup();

	void shutdown();

	void save(std::string filename);

private:

	typedef std::shared_ptr<DeviceBuffer<float>> device_buf_t;

	DoubleBuffer<device_buf_t>* dev_bufs;

	device_buf_t heat;

	Extent2 ext;

	//HSLConverter2 converter = HSLConverter2(max_heat, 0.0f, 0.25f);
	HSLConverter converter = HSLConverter(max_heat);

};



