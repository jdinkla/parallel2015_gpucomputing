/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <iostream>
#include "GPU.h"
#include <vector>
#include <algorithm>
#include <iterator>
#include <map>

using namespace std;

class GPUs
{
public:

	GPUs();

	int get_number_of_gpus() const
	{
		return numberOfGPUs;
	}

	GPU get_gpu(const int i) const
	{
		return gpus[i];
	}

	std::vector<GPU> get_gpus() const
	{
		return gpus;
	}

	// returns all the gpus capable of P2P 
	// (compute capability >= 2 && TCC driver if windows)
	std::vector<GPU> get_p2p_gpus();

	// lazy getter
	std::vector<std::vector<GPU>> get_p2p_cliques()
	{
		if (!p2p_cliques_determined)
		{
			determine_p2p_cliques();
		}
		return p2p_cliques;
	}

	// enables P2P with cudaDeviceEnablePeerAccess()
	void enable_p2p();

	// disables P2P with cudaDeviceEnablePeerAccess()
	void disable_p2p();

	// for looping with "for (auto& gpu : gpus)"
	std::vector<GPU>::const_iterator begin() const
	{
		return gpus.begin();
	}

	// for looping with "for (auto& gpu : gpus)"
	std::vector<GPU>::const_iterator end() const
	{
		return gpus.end();
	}

	// for gpus[1]
	GPU operator[](const int idx) const
	{
		return gpus[idx];
	}

private:

	void determine_gpu_properties();

	// returns a list of P2P "cliques". if all the GPUs are on the same PCIe bus then
	// the result is the same as get_p2p_gpus(). 
	// If there is a QPI link in between it is not.
	// this is a d^2 algorithm. 
	// it assumes, that p2p accessibility is symmetric, i.e. 
	// if cudaDeviceCanAccessPeer(&a, d, e) then cudaDeviceCanAccessPeer(&a, e, d)
	void determine_p2p_cliques();

private:

	// the number of gpus
	int numberOfGPUs;

	// the GPU information
	std::vector<GPU> gpus;

	bool p2p_cliques_determined = false;

	// the P2P 'cliques', see determine_p2p_cliques()
	std::vector<std::vector<GPU>> p2p_cliques;

};