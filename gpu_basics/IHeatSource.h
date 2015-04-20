/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include "Extent2.h"

class IHeatSource
{
public:

	virtual void generate(float* ptr, const Extent2& ext, const int num_sources) = 0;

};
