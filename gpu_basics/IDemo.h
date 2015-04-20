/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include "IHeatSource.h"
#include "RandomHeatSource.h"

class IDemo
{
public:

	virtual ~IDemo()
	{
	}

	virtual void init(const int width, const int height) = 0;

	virtual void render(uchar4* d_image) = 0;

	virtual void update() = 0;

	// at the end of the run, sync all devices
	virtual void synchronize() = 0;

	virtual void cleanup() = 0;

	virtual void shutdown() = 0;

	// save an image of the last frame
	virtual void save(std::string filename) = 0;

	std::shared_ptr<IHeatSource> get_heat_source() const
	{
		return heat_source;
	}

	void set_heat_source(std::shared_ptr<IHeatSource> _heat_source)
	{
		heat_source = _heat_source;
	}

	void set_add_heat(const bool value)
	{
		add_heat = value;
	}

	bool is_add_heat() const
	{
		return add_heat;
	}

private:

	std::shared_ptr<IHeatSource> heat_source = RandomHeatSource::make_shared();

	bool add_heat = true;

};