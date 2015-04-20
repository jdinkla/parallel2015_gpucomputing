/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "HostBuffer.h"

template <typename T, class E = Extent2>
class ManagedBuffer
	: public HostBuffer<T, E>
{

public:

	ManagedBuffer(const E& e);

	virtual ~ManagedBuffer();

	void alloc();

	void free();

	using BaseBuffer<T, E>::ptr;

	using BaseBuffer<T, E>::version;

};