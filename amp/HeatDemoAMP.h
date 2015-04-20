/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include <amp.h>
#include <amp_graphics.h>
#include <cvmarkersobj.h>		// you have 
#include "IDemo.h"
#include <vector>
#include <memory>
#include "UnpinnedBuffer.h"
#include "DoubleBuffer.h"

class HeatDemoAMP
	: public virtual IDemo
{

public:

	HeatDemoAMP();

	virtual ~HeatDemoAMP()
	{
	}

	void init(const int width, const int height);

	void render(uchar4* d_image);

	void update();

	void cleanup();

	void shutdown();

	void save(std::string filename);
	
	void synchronize();

	//void unset_opengl()
	//{
	//	using_opengl = false;
	//}

private:

	typedef std::shared_ptr<UnpinnedBuffer<float>> buf_t;
	typedef concurrency::array_view<float, 2> view_t;

	DoubleBuffer<buf_t>* bufs;
	DoubleBuffer<view_t*>* views;

	buf_t heat_buf;
	concurrency::array_view<const float, 2>* heat_view = nullptr;

	std::shared_ptr<UnpinnedBuffer<uchar4>> image_buf;
	concurrency::array_view<unsigned int, 2>* image_view = nullptr;				// C++ AMP hat kein uchar und kein uchar4! Keine 8-Bit Datentypen

	Extent2 ext;

	concurrency::diagnostic::marker_series mySeries;

	const int accelerator_to_use = 0;
};


