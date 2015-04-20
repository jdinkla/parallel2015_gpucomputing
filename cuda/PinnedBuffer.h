/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "HostBuffer.h"
#include <memory>

template <typename T, class E = Extent2>
class PinnedBuffer
	: public HostBuffer<T, E>
{

public:

	PinnedBuffer(const E& e);

	virtual ~PinnedBuffer();

	void alloc();

	void free();

#ifdef CPP11
	// no copies allowed
	PinnedBuffer(PinnedBuffer<T, E>&) = delete;
#endif

	using BaseBuffer<T, E>::ptr;

	using BaseBuffer<T, E>::version;

	static std::shared_ptr<PinnedBuffer<T, E>> make_shared(const E& e)
	{
		PinnedBuffer<T, E>* ptr = new PinnedBuffer<T, E>(e);
		return std::shared_ptr<PinnedBuffer<T, E>>(ptr);
	}

};