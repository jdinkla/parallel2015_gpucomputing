/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include <cuda_runtime_api.h>
#include <vector>

class GPU
{
public:
	
	GPU(const int _device_id, cudaDeviceProp& _properties);

	// does the GPU support P2P copies?
	bool supports_p2p() const;

	int get_device_id() const
	{
		return device_id;
	}

	cudaDeviceProp get_properties() const
	{
		return properties;
	}

	// add GPU to clique
	void add_to_p2p_clique(GPU gpu);

	std::vector<GPU> get_p2p_clique() const
	{
		return p2p_clique;
	}

	// are p2p copies supported to 'gpu'?
	bool supports_p2p_to(GPU gpu) const;

	// compare just the device_id for equality
	bool operator==(GPU gpu) const
	{
		return device_id == gpu.get_device_id();
	}

	bool operator<(GPU gpu) const
	{
		return device_id < gpu.get_device_id();
	}

private:

	int device_id;			// the id used by cudaSetDevice

	cudaDeviceProp properties;		// the device properties

	std::vector<GPU> p2p_clique;	// the other nodes in the P2P 'clique'
};
