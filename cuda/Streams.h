/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>
#include <vector>
#include <map>

struct Stream
{

	Stream(const int _device, cudaStream_t& _stream)
		: device(_device)
		, stream(_stream)
	{
	}

	cudaStream_t stream;
	int device;
};

class Streams
{
public:

	void add(Stream stream)
	{
		streams.push_back(stream);
		gpuOfStream[stream] = stream.device;
		streamsOfGPU[stream.device].push_back(stream);
	}

private:

	std::vector<Stream> streams;
	std::map<Stream, int> gpuOfStream;
	std::map<int, std::vector<Stream>> streamsOfGPU;
};