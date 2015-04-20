/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "BaseBuffer.h"

template <typename T, class E = Extent2>
class HostBuffer
	: public BaseBuffer<T, E>
{

public:

	HostBuffer(const E& e)
		: BaseBuffer<T, E>(e)
	{
	}

	virtual ~HostBuffer()
	{
	}

};