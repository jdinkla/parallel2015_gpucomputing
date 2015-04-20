/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "GPU.h"

#include <algorithm>

using namespace std;

GPU::GPU(const int _device_id, cudaDeviceProp& _properties)
	: device_id(_device_id)
	, properties(_properties)
{
}

bool GPU::supports_p2p() const
{
#ifdef WINDOWS
	return properties.tccDriver && properties.major >= 2;
#else
	return properties.major >= 2;
#endif
}

void GPU::add_to_p2p_clique(GPU gpu)
{
	p2p_clique.push_back(gpu);
}

bool GPU::supports_p2p_to(GPU gpu) const
{
	return find(p2p_clique.begin(), p2p_clique.end(), gpu) != p2p_clique.end();
}


