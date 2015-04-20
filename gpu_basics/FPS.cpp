#include "FPS.h"
#include <algorithm>

FPS::FPS()
{
	frameCount = 0;
	fpsCount = 0;
}

void FPS::reset()
{
	frameCount = 0;
	fpsCount = fpsLimit - 1;
}

Maybe<float> FPS::update()
{
	Maybe<float> result;
	frameCount++;
	fpsCount++;
	if (fpsCount >= fpsLimit)
	{
		float ifps = 1.f / (timer.get_average() / 1000.f);
		result = Maybe<float>(ifps);
		fpsCount = 0;
		fpsLimit = (int)std::max(1.f, (float)ifps);
		timer.reset();
	}
	return result;
}

void FPS::start_timer()
{
	timer.start();
}

void FPS::stop_timer()
{
	timer.stop();
}

void FPS::reset_timer()
{
	timer.reset();
}

void FPS::delete_timer()
{
	timer.reset();
}
