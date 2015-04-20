/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Maybe.h"
#include "Timer.h"

class FPS
{
public:

	FPS();

	void reset();

	Maybe<float> update();

	void start_timer();

	void stop_timer();

	void reset_timer();

	void delete_timer();

private:

	Timer timer;

	int fpsCount = 0;        // FPS count for averaging
	int fpsLimit = 15;       // FPS limit for sampling
	unsigned int frameCount = 0;

};