/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "ISchedulerInterval.h"
#include <vector>

template <class T>
class SequentialSchedulerInterval
	: public virtual ISchedulerInterval < T >
{

public:

	void sync(const T start, const T end, std::function<void(T)> closure) const
	{
		for (T i = start; i < end; i++)
		{
			closure(i);
		}
	}

};
