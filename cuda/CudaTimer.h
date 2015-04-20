/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <cuda_runtime_api.h>
#include "ITimer.h"

class CudaTimer
	: public ITimer
{
public:
	CudaTimer()
	{
		cudaEventCreate(&start_ev);
		cudaEventCreate(&stop_ev);
	}

	~CudaTimer()
	{
		cudaEventDestroy(start_ev);
		cudaEventDestroy(stop_ev);
	}

	void start()
	{
		cudaEventRecord(start_ev);
	}

	void stop()
	{
		cudaEventRecord(stop_ev);
		cudaEventSynchronize(stop_ev);
	}

	float delta()
	{
		float delta = 0.0f;
		cudaEventElapsedTime(&delta, start_ev, stop_ev);
		return delta;
	}

private:
	cudaEvent_t start_ev, stop_ev;
};

