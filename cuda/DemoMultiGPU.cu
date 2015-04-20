#include "DemoSingleGPU.h"
#include "DeviceBuffer.h"
#include "CudaExecConfig.h"
#include "CudaUtilities.h"
#include <vector>
#include <memory>
#include "HeatUtilities.h"
#include "Mask.h"
#include "StencilHeat.h"
#include "HeatDemoDefs.h"
#include "GPUs.h"
#include "AsyncScheduler.h"
#include "CudaPartition.h"
#include <map>
#include "MaskWithOffset.h"
#include "StencilHeatWithOffset.h"
#include "SequentialScheduler.h"
#include "HeatDemoDefs.h"
#include "ConstantHeatSource.h"
#include "FileUtilities.h"

#undef DEBUG_ASYNC
//#define DEBUG_ASYNC

using namespace std;

using dev_buf_t = DeviceBuffer<float>;
using dev_bufs_t = vector<dev_buf_t*>;
using host_buf_t = UnpinnedBuffer<float>;
using image_t = DeviceBuffer<uchar4>;

const int width = default_width;
const int height = default_height;

#ifdef _DEBUG
const string filename = "CUDA_demo_multi_DEBUG.bmp";
#else
const string filename = "CUDA_demo_multi.bmp";
#endif

// for (auto& p : parts) {
//	  init(p);
// }

inline void do_something_with_the_data(HostBuffer<float>& h)
{
}

host_buf_t* create_heat(const Extent2& ext)
{
	host_buf_t* heat_host = new host_buf_t(ext);
	heat_host->alloc();
	ConstantHeatSource c;
	c.generate(heat_host->get_ptr(), ext, num_heat_sources);
	return heat_host;
}

dev_buf_t* create_and_alloc(const Extent2& ext)
{
	dev_buf_t* d = new DeviceBuffer<float>(ext);
	d->alloc();
	return d;
}

#if defined(WINDOWS)

void init(CudaPartition& partition);								// forward decl
void update(CudaPartition& partition);								// forward decl
void copy_back(CudaPartition& partition, HostBuffer<float>& host);	// forward decl

//IScheduler<CudaPartition>* scheduler = new SequentialScheduler<CudaPartition>();
IScheduler<CudaPartition>* scheduler = new AsyncScheduler<CudaPartition>();

void parallel_for(vector<CudaPartition> ps, function<void(CudaPartition&)> closure)
{
	scheduler->sync(ps, closure);
}

// --------------------------------------------------

vector<CudaPartition> parts;			// Für jede GPU eine Partition
map<GPU, dev_bufs_t> dev_bufs;			// Für jede GPU zwei Device-Buffer
host_buf_t* heat_host;					// Die Wärmequellen auf dem Host
map<GPU, dev_buf_t*> heat_bufs;			// Für jede GPU eine Teilkopie der Wärmequellen
int current = 0;						// Der aktuelle Buffer 0 oder 1

void demo_multi()
{
	GPUs gpus;										// Die GPUs
	Extent2 ext(width, height);						// Der globale Extent
	parts = calc_partitions(gpus, ext);				// Init Partitionen
	heat_host = create_heat(ext);					// Init Wärmequellen

	parallel_for(parts, ::init);					// Init alle GPUs
	current = 1;

	for (int i = 0; i < iterations; i++)			// Loop
	{
		parallel_for(parts, ::update);
		current = !current;
	}

	UnpinnedBuffer<float> host(ext); host.alloc();	// Kopiere D2H
	parallel_for(parts, [&host](CudaPartition& p) { copy_back(p, host); });

	do_something_with_the_data(host);			    // do something with the data

	save_bmp(filename, host, max_heat);

	// Clean up
	for (auto& partition : parts)
	{
		GPU gpu = partition.framework.gpu;
		cudaSetDevice(gpu.get_device_id());
		CUDA::check("cudaSetDevice [cleanup]");
		for (auto& buf : dev_bufs[gpu])
		{
			buf->free();
		}
		heat_bufs[gpu]->free();
	}
	dev_bufs.clear();
	heat_bufs.clear();
	heat_host->free();
	delete heat_host;
	CUDA::reset_all();
}

void init(CudaPartition& partition)
{
	GPU gpu = partition.framework.gpu;
	cudaSetDevice(gpu.get_device_id()); 
	CUDA::check("cudaSetDevice [init]");

	XExtent2& x = partition.data.xext;
	Extent2 glbl = x.get_global_extent();
	Region2 outer = x.get_outer();
	Extent2 outer_extent = x.get_outer().get_extent();

	dev_bufs[gpu].push_back(create_and_alloc(outer_extent));
	dev_bufs[gpu].push_back(create_and_alloc(outer_extent));
	cudaMemset(dev_bufs[gpu][0]->get_ptr(), 0, dev_bufs[gpu][0]->get_size_in_bytes());
	CUDA::check("cudaMemset [init]");

	// create local heat buffer and copy data from host
	dev_buf_t* loc = create_and_alloc(outer_extent);
	heat_bufs[gpu] = loc;
	float* src = &heat_host->get_ptr()[glbl.index(outer.get_offset())];
	cudaMemcpy(loc->get_ptr(), src, loc->get_size_in_bytes(), cudaMemcpyHostToDevice);
	CUDA::check("cudaMemcpy [init]");
}

void update(CudaPartition& partition)
{
	GPU gpu = partition.framework.gpu;
	cudaSetDevice(gpu.get_device_id());
	CUDA::check("cudaSetDevice [update]");

	float* src = dev_bufs[gpu][!current]->get_ptr();
	float* dest = dev_bufs[gpu][current]->get_ptr();

	XExtent2& x = partition.data.xext;
	Extent2 outer_extent = x.get_outer().get_extent();
	Extent2 inner_extent = x.get_inner().get_extent();
	Pos2 offset = x.get_offset_in_local();

	// add_heat()
	mask_2d(outer_extent, *heat_bufs[gpu], *dev_bufs[gpu][!current], Pos2{ 0, 0 });
#ifdef _DEBUG_KERNELS
	cudaDeviceSynchronize(); CUDA::check("mask_2d");
#endif
	// stencil()
	CudaExecConfig cfg_without(inner_extent);
	stencil_heat_2d(cfg_without, *dev_bufs[gpu][!current], 
		*dev_bufs[gpu][current], inner_extent, offset, ct);
#ifdef _DEBUG_KERNELS
	cudaDeviceSynchronize(); CUDA::check("stencil_heat_2d");
#endif

	// Copy halo / ghost cells to other devices
	const int id = partition.partition_id;
	const int src_dev = partition.framework.gpu.get_device_id();
	float* src_base = dev_bufs[gpu][current]->get_ptr();

	if (x.has_high_overlap()) //  && p < num_parts - 1)
	{
		// copy "high inner" to "low overlap", see slides of talk
		// ... from this GPU
		const int hii_idx = x.get_high_inner_start_index();
		float* src = &src_base[hii_idx];
		Region2 hii = x.get_high_inner();
		const int sz = hii.get_extent().get_number_of_elems() * sizeof(float);
		// ... to next GPU
		CudaPartition dst_part = parts[id + 1];
		GPU dst_gpu = dst_part.framework.gpu;
		const int dst_dev = dst_gpu.get_device_id();
		float* dst = dev_bufs[dst_gpu][current]->get_ptr();	// low overlap = (0,0)

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
		CudaPartition dst_part = parts[id - 1];
		GPU dst_gpu = dst_part.framework.gpu;
		const int dst_dev = dst_gpu.get_device_id();
		XExtent2& dst_x = dst_part.data.xext;				// Wichtig: XExtent des Ziels!
		const int hio_idx = dst_x.get_high_overlap_start_index();
		float* dst_base = dev_bufs[dst_gpu][current]->get_ptr();
		float* dst = &dst_base[hio_idx];

#ifdef DEBUG_ASYNC
		cudaMemcpyPeer(dst, dst_dev, src, src_dev, sz);
		CUDA::check("cudaMemcpyPeer[Async] <-");
#else
		cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, sz);
#endif
	}
}

void copy_back(CudaPartition& partition, HostBuffer<float>& host)
{
	GPU gpu = partition.framework.gpu;
	cudaSetDevice(gpu.get_device_id());
	CUDA::check("cudaSetDevice [copy_back]");

	cudaDeviceSynchronize();							// Warte auf Beendigung
	CUDA::check("cudaDeviceSynchronize [copy_back]");

	XExtent2& x = partition.data.xext;
	Extent2 glbl = x.get_global_extent();

	// Source
	dev_buf_t* src_buf = dev_bufs[gpu][!current];
	const int src_idx = glbl.index(x.get_offset_in_local());
	float* src = &(src_buf->get_ptr()[src_idx]);
	// Destination
	const Pos2 offset = x.get_inner_offset();
	const int dst_idx = glbl.index(offset);
	float* dst = &(host.get_ptr()[dst_idx]);

	const size_t sz = x.get_inner().get_extent().get_number_of_elems() * sizeof(float);
	cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost);
	CUDA::check("cudaMemcpy");
}

#endif

// ---------------------------------------------------------------------

#if defined(MAC) || defined(LINUX)

void init(CudaPartition partition);								// forward decl
void update(CudaPartition partition);								// forward decl
void copy_back(CudaPartition partition, HostBuffer<float>& host);	// forward decl

#if defined(MAC)
IScheduler<CudaPartition>* scheduler = new SequentialScheduler<CudaPartition>();
#else
IScheduler<CudaPartition>* scheduler = new AsyncScheduler<CudaPartition>();
#endif

void parallel_for(vector<CudaPartition> ps, function<void(CudaPartition)> closure)
{
	scheduler->sync(ps, closure);
}

// --------------------------------------------------

vector<CudaPartition> parts;			// Für jede GPU eine Partition
map<GPU, dev_bufs_t> dev_bufs;			// Für jede GPU zwei Device-Buffer
host_buf_t* heat_host;					// Die Wärmequellen auf dem Host
map<GPU, dev_buf_t*> heat_bufs;			// Für jede GPU eine Teilkopie der Wärmequellen
int current = 0;						// Der aktuelle Buffer 0 oder 1

void demo_multi()
{
	GPUs gpus;										// Die GPUs
	Extent2 ext(width, height);						// Der globale Extent
	parts = calc_partitions(gpus, ext);				// Init Partitionen
	heat_host = create_heat(ext);					// Init Wärmequellen

	parallel_for(parts, ::init);					// Init alle GPUs
	current = 1;

	for (int i = 0; i < iterations; i++)			// Loop
	{
		parallel_for(parts, ::update);
		current = !current;
	}

	UnpinnedBuffer<float> host(ext); host.alloc();	// Kopiere D2H
	parallel_for(parts, [&host](CudaPartition p) { copy_back(p, host); });

	do_something_with_the_data(host);			    // do something with the data

	save_bmp(filename, host, max_heat);

	// Clean up
	for (auto& partition : parts)
	{
		GPU gpu = partition.framework.gpu;
		cudaSetDevice(gpu.get_device_id());
		CUDA::check("cudaSetDevice [cleanup]");
		for (auto& buf : dev_bufs[gpu])
		{
			buf->free();
		}
		heat_bufs[gpu]->free();
	}
	dev_bufs.clear();
	heat_bufs.clear();
	heat_host->free();
	delete heat_host;
	CUDA::reset_all();
}

void init(CudaPartition partition)
{
	GPU gpu = partition.framework.gpu;
	cudaSetDevice(gpu.get_device_id()); 
	CUDA::check("cudaSetDevice [init]");

	XExtent2& x = partition.data.xext;
	Extent2 glbl = x.get_global_extent();
	Region2 outer = x.get_outer();
	Extent2 outer_extent = x.get_outer().get_extent();

	dev_bufs[gpu].push_back(create_and_alloc(outer_extent));
	dev_bufs[gpu].push_back(create_and_alloc(outer_extent));
	cudaMemset(dev_bufs[gpu][0]->get_ptr(), 0, dev_bufs[gpu][0]->get_size_in_bytes());
	CUDA::check("cudaMemset [init]");

	// create local heat buffer and copy data from host
	dev_buf_t* loc = create_and_alloc(outer_extent);
	heat_bufs[gpu] = loc;
	float* src = &heat_host->get_ptr()[glbl.index(outer.get_offset())];
	cudaMemcpy(loc->get_ptr(), src, loc->get_size_in_bytes(), cudaMemcpyHostToDevice);
	CUDA::check("cudaMemcpy [init]");
}

void update(CudaPartition partition)
{
	GPU gpu = partition.framework.gpu;
	cudaSetDevice(gpu.get_device_id());
	CUDA::check("cudaSetDevice [update]");

	float* src = dev_bufs[gpu][!current]->get_ptr();
	float* dest = dev_bufs[gpu][current]->get_ptr();

	XExtent2& x = partition.data.xext;
	Extent2 outer_extent = x.get_outer().get_extent();
	Extent2 inner_extent = x.get_inner().get_extent();
	Pos2 offset = x.get_offset_in_local();

	// add_heat()
	mask_2d(outer_extent, *heat_bufs[gpu], *dev_bufs[gpu][!current], Pos2{ 0, 0 });
#ifdef _DEBUG_KERNELS
	cudaDeviceSynchronize(); CUDA::check("mask_2d");
#endif
	// stencil()
	CudaExecConfig cfg_without(inner_extent);
	stencil_heat_2d(cfg_without, *dev_bufs[gpu][!current], 
		*dev_bufs[gpu][current], inner_extent, offset, ct);
#ifdef _DEBUG_KERNELS
	cudaDeviceSynchronize(); CUDA::check("stencil_heat_2d");
#endif

	// Copy halo / ghost cells to other devices
	const int id = partition.partition_id;
	const int src_dev = partition.framework.gpu.get_device_id();
	float* src_base = dev_bufs[gpu][current]->get_ptr();

	if (x.has_high_overlap()) //  && p < num_parts - 1)
	{
		// copy "high inner" to "low overlap", see slides of talk
		// ... from this GPU
		const int hii_idx = x.get_high_inner_start_index();
		float* src = &src_base[hii_idx];
		Region2 hii = x.get_high_inner();
		const int sz = hii.get_extent().get_number_of_elems() * sizeof(float);
		// ... to next GPU
		CudaPartition dst_part = parts[id + 1];
		GPU dst_gpu = dst_part.framework.gpu;
		const int dst_dev = dst_gpu.get_device_id();
		float* dst = dev_bufs[dst_gpu][current]->get_ptr();	// low overlap = (0,0)

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
		CudaPartition dst_part = parts[id - 1];
		GPU dst_gpu = dst_part.framework.gpu;
		const int dst_dev = dst_gpu.get_device_id();
		XExtent2& dst_x = dst_part.data.xext;				// Wichtig: XExtent des Ziels!
		const int hio_idx = dst_x.get_high_overlap_start_index();
		float* dst_base = dev_bufs[dst_gpu][current]->get_ptr();
		float* dst = &dst_base[hio_idx];

#ifdef DEBUG_ASYNC
		cudaMemcpyPeer(dst, dst_dev, src, src_dev, sz);
		CUDA::check("cudaMemcpyPeer[Async] <-");
#else
		cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, sz);
#endif
	}
}

void copy_back(CudaPartition partition, HostBuffer<float>& host)
{
	GPU gpu = partition.framework.gpu;
	cudaSetDevice(gpu.get_device_id());
	CUDA::check("cudaSetDevice [copy_back]");

	cudaDeviceSynchronize();							// Warte auf Beendigung
	CUDA::check("cudaDeviceSynchronize [copy_back]");

	XExtent2& x = partition.data.xext;
	Extent2 glbl = x.get_global_extent();

	// Source
	dev_buf_t* src_buf = dev_bufs[gpu][!current];
	const int src_idx = glbl.index(x.get_offset_in_local());
	float* src = &(src_buf->get_ptr()[src_idx]);
	// Destination
	const Pos2 offset = x.get_inner_offset();
	const int dst_idx = glbl.index(offset);
	float* dst = &(host.get_ptr()[dst_idx]);

	const size_t sz = x.get_inner().get_extent().get_number_of_elems() * sizeof(float);
	cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost);
	CUDA::check("cudaMemcpy");
}

#endif


#undef FOLIE 
//#define FOLIE 

#ifdef FOLIE

void x()
{
	
using dev_buf_t = DeviceBuffer<float>;
using dev_bufs_t = vector<dev_buf_t*>;
using host_buf_t = UnpinnedBuffer<float>;
vector<CudaPartition> parts;		// Für jede GPU eine Partition
map<GPU, dev_bufs_t> dev_bufs;		// Für jede GPU zwei Device-Buffer
host_buf_t* heat_host;				// Die Wärmequellen auf dem Host
map<GPU, dev_buf_t*> heat_bufs;		// Für jede GPU Teil der Wärmequellen
int current = 0;					// Der aktuelle Buffer 0 oder 1
GPUs gpus;							// Die GPUs
Extent2 ext(width, height);			// Der globale Extent
parts = calc_partitions(gpus, ext);	// Init Partitionen
heat_host = create_heat(ext);		// Init Wärmequellen
parallel_for(parts, init);	    	// Init alle GPUs
current = 1;
for (int i = 0; i < iterations; i++)			// Loop
{
	parallel_for(parts, ::update);
	current = !current;
}
UnpinnedBuffer<float> host(ext); host.alloc();	// Kopiere D2H
parallel_for(parts, [&host](CudaPartition& p) { ::copy_back(p, host); });
do_something_with_the_data(host);			    // do something with the data

}
#endif
