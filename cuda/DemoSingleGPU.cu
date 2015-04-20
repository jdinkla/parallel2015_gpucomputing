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
#include "PinnedBuffer.h"
#include "FileUtilities.h"
#include "Convert.h"
#include "MapHost.h"
#include "ConstantHeatSource.h"

using namespace std;

const int w = small_width;
const int h = small_height;
Extent2 ext(w, h);

typedef UnpinnedBuffer<float> buf_t;

buf_t* create_alloced()
{
	buf_t* d = new UnpinnedBuffer<float>(ext);
	d->alloc();
	return d;
}

void init_heat(PinnedBuffer<float>& heat)
{
	heat.alloc();									// buffer heat
	ConstantHeatSource c;
	c.generate(heat.get_ptr(), ext, num_heat_sources);
}

inline void do_something_with_the_data(HostBuffer<float>& h)
{
}

void demo_single()
{
	// Init
	typedef DeviceBuffer<float>* dev_buf_t;
	Extent2 ext(w, h); CudaExecConfig cfg(ext);
	auto create = [&ext]() { 
		auto d = new DeviceBuffer<float>(ext);
		d->alloc();
		return d;
	};

	vector<dev_buf_t> dblbuf{ create(), create() };			// the double buffer
	cudaMemset(dblbuf[0]->get_ptr(), 0, dblbuf[0]->get_size_in_bytes()); // set to 0
	CUDA::check("cudaMemset");

	PinnedBuffer<float> heat_host(ext); init_heat(heat_host);	// generate heat sources
	dev_buf_t heat = create(); heat->copy_from(heat_host);	heat_host.free();

	dev_buf_t previous = dblbuf[0];							// we calculated buffer 0
	dev_buf_t current = dblbuf[1];							// this is the next one to calc
	for (int i = 0; i < iterations; i++)
	{
		mask_2d<float>(cfg, *heat, *previous);				// add_heat();
		stencil_heat_2d(cfg, *previous, *current, ct);		// stencil();
		std::swap(previous, current);
	}
	cudaDeviceSynchronize();
	CUDA::check("sync [update]");

	PinnedBuffer<float> host(ext);host.alloc();				// copy result back to host
	cudaMemcpy(host.get_ptr(), previous->get_ptr(), 
		previous->get_size_in_bytes(), cudaMemcpyDeviceToHost);
	CUDA::check("cudaMemcpy [update]");
	do_something_with_the_data(host);						// do something with the data

	// save to bmp
#ifdef _DEBUG
	save_bmp("CUDA_demo_single_DEBUG.bmp", host, max_heat);
#else
	save_bmp("CUDA_demo_single.bmp", host, max_heat);
#endif

	// Clean up
	heat->free();
	dblbuf[0]->free();
	dblbuf[1]->free();
	dblbuf.clear();
	host.free();

	CUDA::reset_all();
}

