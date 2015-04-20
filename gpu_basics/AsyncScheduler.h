/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "IScheduler.h"
#include <future>
#include <vector>

template <class T>
class AsyncScheduler
	: public virtual IScheduler < T >
{

public:

	void sync(std::vector<T> partitions, std::function<void(T)> closure) const
	{
		// Start all
		std::vector<decltype(async(closure, partitions[0]))> futures;
		for (auto& partition : partitions)
		{
			auto f = std::async(std::launch::async, closure, partition);
			futures.push_back(move(f));
		}
		// Sync
		for (auto& f : futures)
		{
			f.get();
		}
	}

};

