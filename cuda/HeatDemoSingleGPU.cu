/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "HeatDemoSingleGPU.h"
#include <iostream>
#include "Logger.h"
#include "CudaUtilities.h"
#include "CudaExecConfig.h"
#include "Map.h"
#include "Convert.h"
#include "StencilHeat.h"
#include "Stencil.h"
#include "HeatUtilities.h"
#include "Map2.h"
#include "Mask.h"
#include "HeatDemoDefs.h"
#include "NvtxUtilities.h"
#include "FileUtilities.h"

using namespace std;

#define USE_RENDER_HOST_BUFFER

HeatDemoSingleGPU::HeatDemoSingleGPU()
{
	converter.init();
}

void HeatDemoSingleGPU::init(const int _width, const int _height)
{
	DEBUG("init"); START_RANGE("init");

	ext = Extent2(_width, _height);

	// init_heat();
	PinnedBuffer<float> heat_host(ext);
	heat_host.alloc();
	const int num_points = (int)(ext.get_number_of_elems() * ratio_of_heat_sources);
	get_heat_source()->generate(heat_host.get_ptr(), ext, num_points);
	heat = device_buf_t(new DeviceBuffer<float>(ext));			// copy to device
	heat->alloc();
	heat->copy_from(heat_host);
	heat_host.free();

	device_buf_t d0 = DeviceBuffer<float>::make_shared(ext);
	d0->alloc();

	device_buf_t d1 = DeviceBuffer<float>::make_shared(ext);
	d1->alloc();

	dev_bufs = new DoubleBuffer<device_buf_t>(d0, d1);

	// fill d0
	cudaMemset(d0->get_ptr(), 0, d0->get_size_in_bytes());
	CUDA::check("cudaMemset");

	// we calculated buffer 0, so the next 'one' is 1
	dev_bufs->swap();
	END_RANGE;
}

void HeatDemoSingleGPU::render(uchar4* d_image)
{
	DEBUG("render"); START_RANGE("render");

#ifndef USE_RENDER_HOST_BUFFER

	CudaExecConfig cfg(ext);
	map_2d(cfg,
		dev_bufs->get_previous()->get_ptr(),
		d_image,
		ext,
		converter 
		);

	cudaDeviceSynchronize();
	CUDA::check("render_cuda");
#else
	// render on device ...
	DeviceBuffer<uchar4> image(ext);
	image.alloc();

	CudaExecConfig cfg(ext);
	map_2d(cfg,
		dev_bufs->get_previous()->get_ptr(),
		image.get_ptr(),
		ext,
		converter
		);

	cudaDeviceSynchronize();
	CUDA::check("render_cuda");

	// and copy to host ...
	cudaMemcpy(d_image, image.get_ptr(), image.get_size_in_bytes(), cudaMemcpyDeviceToHost);
	CUDA::check("cudaMemcpy");
#endif

	END_RANGE;
}

void HeatDemoSingleGPU::update()
{
	DEBUG("update"); START_RANGE("update");

	DeviceBuffer<float>& previous = *dev_bufs->get_previous();
	DeviceBuffer<float>& current = *dev_bufs->get_current();
	DeviceBuffer<float>& heat_buf = *heat.get();

	CudaExecConfig cfg(ext);
	if (is_add_heat())
	{
		mask_2d<float>(cfg, heat_buf, previous);		// add_heat();
	}
	stencil_heat_2d(cfg, previous, current, ct);		// stencil();

	dev_bufs->swap();
	END_RANGE;
}

void HeatDemoSingleGPU::synchronize()
{
	cudaDeviceSynchronize();
	//CUDA::sync_all();	 produziert hier päter einen Fehler. TODO mit dem richtigen 7.0 Release checken
}

void HeatDemoSingleGPU::save(std::string filename)
{
	ManagedBuffer<uchar4> image(ext);
	image.alloc();
	DeviceBuffer<float>& previous = *dev_bufs->get_previous();

	const CudaExecConfig cfg(ext);
	map_2d(cfg,
		previous.get_ptr(),
		image.get_ptr(),
		ext,
		converter
		);
	cudaDeviceSynchronize();
	CUDA::check("map_2d [save]");

	save_bmp(filename, image.get_ptr(), ext);
	image.free();
}

void HeatDemoSingleGPU::cleanup()
{
	DEBUG("cleanup"); START_RANGE("cleanup");
	heat->free();
	dev_bufs->get_current()->free();
	dev_bufs->get_previous()->free();
	delete dev_bufs;
	END_RANGE;
}

void HeatDemoSingleGPU::shutdown()
{
	START_RANGE("shutdown");
	converter.shutdown();
	END_RANGE;
	CUDA::reset_all();
}
