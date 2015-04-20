/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <functional>
#include <vector>

template <class T>
class IScheduler
{

public:

	virtual void sync(std::vector<T> partitions, std::function<void(T)> closure) const = 0 ;

};
