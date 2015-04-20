/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "ISchedulerInterval.h"
#include <future>
#include <vector>
#include "PartitionUtilities.h"

template <class T>
class AsyncSchedulerInterval
	: public virtual ISchedulerInterval < T >
{

public:

	AsyncSchedulerInterval()
		: num_parts((int)std::thread::hardware_concurrency()) // find out the number of cores
	{

	}

	AsyncSchedulerInterval(const int _num_parts)
		: num_parts(_num_parts)
	{
	}

	bool is_small_interval(const T start, const T end) const
	{
		return end - start < 10;
	}

	// subdivides [start, end] into #cores intervals
	void sync(const T start, const T end, std::function<void(T)> closure) const
	{
		auto loop = [closure](interval_t in)
		{
			for (T i = in.first; i <= in.second; i++)
			{
				closure(i);
			}
		};
		if (num_parts == 0 || is_small_interval(start, end))
		{
			loop(interval_t{ start, end - 1 });					// do sequential
		}
		else
		{
			partition_t ps = partition((int)start, (int)end - 1, num_parts);
			std::vector<decltype(std::async(closure, start))> futures;
			for (auto& p : ps)
			{
				auto f = std::async(std::launch::async, loop, p);
				futures.push_back(move(f));
			}
			// Sync
			for (auto& f : futures)
			{
				f.get();
			}
		}
	}

private:

	const int num_parts;

};

