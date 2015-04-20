/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "IScheduler.h"
#include <vector>

template <class T>
class SequentialScheduler
	: public virtual IScheduler < T >
{

public:

	void sync(std::vector<T> partitions, std::function<void(T)> closure) const
	{
		for (auto& partition : partitions)
		{
			closure(partition);
		}
	}

};

