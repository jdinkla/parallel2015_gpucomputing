/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "HeatDemoMultiGPU.h"
#include <iostream>
#include "Logger.h"
#include "CudaUtilities.h"
#include "CudaExecConfig.h"
#include "Map.h"
#include "Convert.h"
#include "StencilHeat.h"
#include "Partition.h"
#include "PartitionUtilities.h"
#include "HeatDemoDefs.h"
#include "HeatUtilities.h"
#include "MaskWithOffset.h"
#include "StencilHeatWithOffset.h"
#include "SequentialScheduler.h"
#include "AsyncScheduler.h"
#include "NvtxUtilities.h"

using namespace std;

#if defined(MAC) || defined(LINUX)
template <typename I, typename O, class F>
void map_2d(
	const CudaExecConfig& cnf,
	const I* src,
	O* dst,
	Extent2 e,
	const F functor)
{
	map_2d_kernel << <cnf.get_grid(), cnf.get_block(), 0, cnf.get_stream() >> >(src, dst, e, functor);
}
#endif

//#define DEBUG_ASYNC 1

HeatDemoMultiGPU::HeatDemoMultiGPU()
#ifdef MAC
	: scheduler(new SequentialScheduler<CudaPartition>())
#else
	: scheduler(new AsyncScheduler<CudaPartition>())
#endif	
{
	gpus.enable_p2p();
}

HeatDemoMultiGPU::device_buf_t create_and_alloc(Extent2& ext)
{
	HeatDemoMultiGPU::device_buf_t d = make_shared<DeviceBuffer<float>>(ext);
	d->alloc();
	return d;
}

void HeatDemoMultiGPU::init(CudaPartition& partition)
{
	DEBUG("init(p)"); START_RANGE("init(p)");
	try
	{
		GPU gpu = partition.framework.gpu;
		cudaSetDevice(gpu.get_device_id());
		CUDA::check("cudaSetDevice [init]");

		XExtent2& x = partition.data.xext;
		Region2 outer = x.get_outer();
		Extent2 outer_extent = outer.get_extent();
		Region2 inner = x.get_inner();
		Extent2 inner_extent = inner.get_extent();

		dev_bufs[gpu].push_back(create_and_alloc(outer_extent));
		dev_bufs[gpu].push_back(create_and_alloc(outer_extent));

		// fill d0
		device_buf_t dev = dev_bufs[gpu][0];
		cudaMemset(dev->get_ptr(), 0, dev->get_size_in_bytes());
		CUDA::check("cudaMemset [init]");

		// create local heat buffer and copy data from host
		device_buf_t local_heat = create_and_alloc(outer_extent);
		float* src = &heat_host->get_ptr()[ext.index(outer.get_offset())];
		const size_t sz = local_heat->get_size_in_bytes();
		cudaMemcpy(local_heat->get_ptr(), src, sz, cudaMemcpyHostToDevice);
		CUDA::check("cudaMemcpy [init]");
		heat_bufs[gpu] = local_heat;

		// local image buffer
		image_t local_image = make_shared<DeviceBuffer<uchar4 >> (inner_extent);
		local_image->alloc();
		image_bufs[gpu] = local_image;
	}
	catch (std::runtime_error e)
	{
		cerr << "ERROR: " << e.what() << endl;
	}
	END_RANGE;
}

void HeatDemoMultiGPU::init(const int _width, const int _height)
{
	DEBUG("init"); START_RANGE("init");

	if (using_opengl)
	{
		opengl_device = CUDA::get_opengl_device();			// do not call this before GL was initialized
	}

	ext = Extent2(_width, _height);
	partitions = calc_partitions(gpus, ext);

	// init_heat();
	heat_host = PinnedBuffer < float >::make_shared(ext);
	heat_host->alloc();
	const int num_points = (int)(ext.get_number_of_elems() * ratio_of_heat_sources);
	get_heat_source()->generate(heat_host->get_ptr(), ext, num_points);

#if defined(MAC) || defined(LINUX)
	scheduler->sync(partitions, [this](CudaPartition partition) { this->init(partition); });
#else
	scheduler->sync(partitions, [this](CudaPartition& partition) { this->init(partition); });
#endif

	current = 1;
	END_RANGE;
}

void HeatDemoMultiGPU::render(CudaPartition& partition, uchar4* d_image)
{
	DEBUG("render(p)"); START_RANGE("render(p)");
	try
	{
		GPU gpu = partition.framework.gpu;
		cudaSetDevice(gpu.get_device_id());
		CUDA::check("cudaSetDevice [render(p)]");

		XExtent2 x = partition.data.xext;
		Region2 inner = x.get_inner();
		Pos2 global_offset = inner.get_offset();

		// render the image on the current GPU
		const CudaExecConfig cfg(image_bufs[gpu]->get_extent());
		map_2d(cfg,
			dev_bufs[gpu][!current]->get_ptr(),
			image_bufs[gpu]->get_ptr(),
			image_bufs[gpu]->get_extent(),
			converter
			);

		const int dst_dev = opengl_device;
		uchar4* dst = &d_image[ext.index(global_offset)];

		const int src_dev = gpu.get_device_id();
		uchar4* src = image_bufs[gpu]->get_ptr();

		const size_t sz = image_bufs[gpu]->get_size_in_bytes();

		if (dst_dev == src_dev)
		{
			cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToDevice);
			CUDA::check("cudaMemcpy [render(p)]");
		}
		else
		{
			cudaMemcpyPeer(dst, dst_dev, src, src_dev, sz);
			CUDA::check("cudaMemcpyPeer [render(p)]");
		}
	}
	catch (std::runtime_error e)
	{
		cerr << "ERROR: " << e.what() << endl;
	}
	END_RANGE;
}


void HeatDemoMultiGPU::render(uchar4* d_image)
{
	DEBUG("render"); START_RANGE("render");

#if defined(MAC) || defined(LINUX)
	scheduler->sync(partitions, [this, &d_image](CudaPartition partition)
	{
		this->render(partition, d_image);
	});
#else
	scheduler->sync(partitions, [this, &d_image](CudaPartition& partition)
	{
		this->render(partition, d_image);
	});
#endif	

	END_RANGE;
}

void HeatDemoMultiGPU::update(CudaPartition& partition)
{
	DEBUG("update(p)"); START_RANGE("update(p)");
	GPU gpu = partition.framework.gpu;
	cudaSetDevice(gpu.get_device_id());
	CUDA::check("cudaSetDevice [update(p)]");

	float* src = dev_bufs[gpu][!current]->get_ptr();
	float* dest = dev_bufs[gpu][current]->get_ptr();

	XExtent2& x = partition.data.xext;
	Extent2 outer_extent = x.get_outer().get_extent();
	Extent2 inner_extent = x.get_inner().get_extent();
	Pos2 offset = x.get_offset_in_local();

	// add_heat()
	if (is_add_heat())
	{
		mask_2d(outer_extent, heat_bufs[gpu], dev_bufs[gpu][!current], Pos2{ 0, 0 });
#ifdef _DEBUG_KERNELS
		cudaDeviceSynchronize();
		CUDA::check("mask_2d");
#endif
	}

	// stencil()
	CudaExecConfig cfg_without(inner_extent);
	stencil_heat_2d(cfg_without, dev_bufs[gpu][!current], dev_bufs[gpu][current], inner_extent, offset, ct);
#ifdef _DEBUG_KERNELS
	cudaDeviceSynchronize();
	CUDA::check("stencil_heat_2d");
#endif

	// Copy halo / ghost cells to other devices
	const int copy_buf = current;

	const int id = partition.partition_id;
	const int src_dev = partition.framework.gpu.get_device_id();
	float* src_base = dev_bufs[gpu][copy_buf]->get_ptr();

	if (x.has_high_overlap()) //  && p < num_parts - 1)
	{
		// copy "high inner" to "low overlap", see slides of talk
		// ... from this GPU
		const int hii_idx = x.get_high_inner_start_index();
		float* src = &src_base[hii_idx];
		Region2 hii = x.get_high_inner();
		const int sz = hii.get_extent().get_number_of_elems() * sizeof(float);
		// ... to next GPU
		CudaPartition dst_part = partitions[id + 1];
		GPU dst_gpu = dst_part.framework.gpu;
		const int dst_dev = dst_gpu.get_device_id();
		float* dst = dev_bufs[dst_gpu][copy_buf]->get_ptr();	// low overlap = (0,0)

#ifdef DEBUG_ASYNC
		cudaMemcpyPeer(dst, dst_dev, src, src_dev, sz);
		CUDA::check("cudaMemcpyPeer[Async] ->");
#else
		cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, sz);
#endif
	}

	if (x.has_low_overlap())
	{
		// copy "low inner" to "high overlap", see slides of talk
		// ... from this GPU 
		const int loi_idx = x.get_low_inner_start_index();
		float* src = &src_base[loi_idx];
		Region2 loi = x.get_low_inner();
		const int sz = loi.get_extent().get_number_of_elems() * sizeof(float);
		// ... to prev GPU
		CudaPartition dst_part = partitions[id - 1];
		GPU dst_gpu = dst_part.framework.gpu;
		const int dst_dev = dst_gpu.get_device_id();
		XExtent2& dst_x = dst_part.data.xext;				// Wichtig: XExtent des Ziels!
		const int hio_idx = dst_x.get_high_overlap_start_index();
		float* dst_base = dev_bufs[dst_gpu][copy_buf]->get_ptr();
		float* dst = &dst_base[hio_idx];

#ifdef DEBUG_ASYNC
		cudaMemcpyPeer(dst, dst_dev, src, src_dev, sz);
		CUDA::check("cudaMemcpyPeer[Async] <-");
#else
		cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, sz);
#endif
	}

	END_RANGE;
}

void HeatDemoMultiGPU::update()
{
	DEBUG("update"); START_RANGE("update");
#if defined(MAC) || defined(LINUX)
	scheduler->sync(partitions, [this](CudaPartition partition)
	{
		update(partition);
	});
#else	
	scheduler->sync(partitions, [this](CudaPartition& partition)
	{
		update(partition);
	});
#endif	
	current = !current;
	END_RANGE;
}

void HeatDemoMultiGPU::synchronize()
{
	CUDA::sync_all();
}

void HeatDemoMultiGPU::cleanup()
{
	DEBUG("cleanup"); START_RANGE("cleanup");
	for (auto& partition : partitions)
	{
		GPU gpu = partition.framework.gpu;
		cudaSetDevice(gpu.get_device_id());
		CUDA::check("cudaSetDevice [cleanup]");
		for (auto& buf : dev_bufs[gpu])
		{
			buf->free();
		}
		heat_bufs[gpu]->free();
		image_bufs[gpu]->free();
	}
	dev_bufs.clear();
	heat_bufs.clear();
	image_bufs.clear();
	heat_host->free();
	END_RANGE;
}

void HeatDemoMultiGPU::shutdown()
{
	START_RANGE("shutdown");
	CUDA::reset_all();
	END_RANGE;
}
