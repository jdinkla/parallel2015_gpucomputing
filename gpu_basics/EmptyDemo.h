/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "IDemo.h"
#include "Logger.h"

// used for testing
class EmptyDemo
	: public virtual IDemo
{

public:

	void init(const int width, const int height)
	{
		DEBUG("init");
	}

	void render(uchar4* d_image)
	{
		DEBUG("render");
	}

	void update()
	{
		DEBUG("update");
	}

	void cleanup()
	{
		DEBUG("cleanup");
	}

	void shutdown()
	{
		DEBUG("shutdown");
	}

};



