/*
 * Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#pragma once

#include <chrono>
#include <string>
#include <functional>
#include "ITimer.h"

class Timer
	: public virtual ITimer
{
public:

	using clock = std::chrono::steady_clock;

	Timer()
	{
	}

	void start()
	{
		start_val = clock::now();
	}

	void stop()
	{
		stop_val = clock::now();
		total += delta();
		count++;
	}

	float delta()
	{
		last_delta = (float)get_duration().count();
		return last_delta;
	}

	std::chrono::milliseconds get_duration()
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(stop_val - start_val);
	}

	float get_average()
	{
		return (count > 0) ? (total / count) : 0.0f;
	}

	void reset()
	{
		last_delta = 0.0f;
		total = 0.0f;
		count = 0;
	}

private:

	clock::time_point start_val;

	clock::time_point stop_val;

	float last_delta = 0.0f;

	float total = 0.0f;

	int count = 0;

};

// formats the time to YYYY-MM-DD HH:mm:SS.MS
std::string format_time(std::chrono::system_clock::time_point tp);

// formats the current time to YYYY-MM-DD HH:mm:SS.MS
std::string format_time();

void check_chrono();

