/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "GPUs.h"
#include "CudaUtilities.h"

using namespace std;

GPUs::GPUs()
{
	numberOfGPUs = CUDA::get_number_of_gpus();
	determine_gpu_properties();
}

void GPUs::determine_gpu_properties()
{
	for (int d = 0; d < numberOfGPUs; d++)
	{
		// find properties
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, d);
		CUDA::check("cudaGetDeviceProperties");
		// add to vector
		GPU gpu(d, prop);
		gpus.push_back(gpu);
	}
}

std::vector<GPU> GPUs::get_p2p_gpus()
{
	std::vector<GPU> result;
	for (auto& gpu : gpus)
	{
		if (gpu.supports_p2p())
		{
			result.push_back(gpu);
		}
	}
	return result;
}

// idea taken from the CUDA sample p2pBandwidthLatencyTest.cu
void GPUs::determine_p2p_cliques()
{
	if (!p2p_cliques_determined)
	{
		p2p_cliques.resize(0);
		vector<bool> done(numberOfGPUs, false);
		for (int d = 0; d < numberOfGPUs; d++)
		{
			if (!done[d])
			{
				//create a new clique with GPU d
				vector<GPU> newClique;
				done[d] = true;
				newClique.push_back(gpus[d]);
				// are the other devices available from this device?
				for (int e = d + 1; e < numberOfGPUs; e++)
				{
					int access;
					cudaDeviceCanAccessPeer(&access, d, e);
					CUDA::check("cudaDeviceCanAccessPeer");
					if (access)
					{
						newClique.push_back(gpus[e]);
						gpus[e].add_to_p2p_clique(gpus[d]);
						gpus[d].add_to_p2p_clique(gpus[e]);
						done[e] = true;
					}
				}
				p2p_cliques.push_back(newClique);
			}
		}
		p2p_cliques_determined = true;
	}
}

void GPUs::enable_p2p()
{
	determine_p2p_cliques();
	for (auto& clique : p2p_cliques)
	{
		const int num = (int) clique.size();
		for (int d = 0; d < num; d++)
		{
			cudaSetDevice(d);
			for (int e = 0; e < num && d != e; e++)
			{
				cudaDeviceEnablePeerAccess(e, 0);
				CUDA::check("cudaDeviceEnablePeerAccess");
			}
		}
	}
}

void GPUs::disable_p2p()
{
	determine_p2p_cliques();
	for (auto& clique : p2p_cliques)
	{
		const int num = (int) clique.size();
		for (int d = 0; d < num; d++)
		{
			cudaSetDevice(d);
			for (int e = 0; e < num && d != e; e++)
			{
				cudaDeviceDisablePeerAccess(e);
				CUDA::check("cudaDeviceDisablePeerAccess");
			}
		}
	}
}
