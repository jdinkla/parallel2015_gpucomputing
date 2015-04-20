/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "IScheduler.h"
#include <thread>
#include <vector>

template <class T>
class ThreadScheduler
	: public virtual IScheduler < T >
{

public:

	void sync(std::vector<T> partitions, std::function<void(T)> closure) const
	{
		// Start all
		std::vector<thread*> threads;
		for (auto& partition : partitions)
		{
			thread* t = new thread(closure, partition);
			threads.push_back(t);
		}
		// Sync
		for (auto t : threads)
		{
			t->join();
		}
		threads.clear();
	}

};

