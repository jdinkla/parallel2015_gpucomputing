/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "ISchedulerInterval.h"
#include <thread>
#include <vector>

template <class T>
class ThreadSchedulerInterval
	: public virtual ISchedulerInterval < T >
{

public:

	void sync(const T start, const T end, std::function<void(T)> closure) const
	{
		std::vector<thread*> threads;
		for (T i = start; i < end; i++)
		{
			thread* t = new thread(closure, i);
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
