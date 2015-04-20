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
#include <map>
#include "DeviceBuffer.h"
#include "Converter.h"
#include "GPUs.h"
#include "XExtent2.h"
#include "Partition.h"
#include "CudaPartition.h"
#include "IScheduler.h"
#include "HeatDemoDefs.h"

class HeatDemoMultiGPU
	: public virtual IDemo
{

public:

	HeatDemoMultiGPU();

	virtual ~HeatDemoMultiGPU()
	{
	}

	void init(const int width, const int height);

	void render(uchar4* d_image);

	void update();

	void synchronize();

	void cleanup();

	void shutdown();

	void save(std::string filename)
	{
	}

	void unset_opengl()
	{
		using_opengl = false;
	}

	void set_scheduler(IScheduler<CudaPartition>* _scheduler)
	{
		scheduler = _scheduler;
	}

	// the data 
	using device_buf_t = std::shared_ptr < DeviceBuffer<float> > ;
	using device_bufs_t = std::vector < device_buf_t > ;
	using host_buf_t = std::shared_ptr < PinnedBuffer<float> >;
	using image_t = std::shared_ptr < DeviceBuffer<uchar4> > ;

private:

	void init(CudaPartition& partition);

	void update(CudaPartition& partition);

	void render(CudaPartition& partition, uchar4* d_image);

	std::vector<CudaPartition> partitions;

	std::map<GPU, device_bufs_t> dev_bufs;				// the device buffers, 2 for each GPU
	std::map<GPU, image_t> image_bufs;					// the image buffers, 1 for each GPU
	host_buf_t heat_host;								// the global heat sources
	std::map<GPU, device_buf_t> heat_bufs;				// the heat buffers, 1 for each GPU

	int current = 0;

	Extent2 ext;

	GPUs gpus;

	bool using_opengl = true;

	int opengl_device;

	IScheduler<CudaPartition>* scheduler;

	HSLConverter converter = HSLConverter(max_heat);

};
