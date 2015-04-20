/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <functional>
#include <vector>

template <class T>
class ISchedulerInterval
{

public:

	virtual void sync(const T start, const T end, std::function<void(T)> closure) const = 0;

};
