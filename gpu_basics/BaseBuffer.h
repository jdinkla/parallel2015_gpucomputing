/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include "Extent2.h"
#include <iostream>
#include <sstream>
#include <algorithm>

template <typename T, class E = Extent2>
class BaseBuffer
{

public:

	BaseBuffer(const E& e)
		: ptr(nullptr)
		, extent(e)
		, version(-1)
	{
	}

	virtual ~BaseBuffer()
	{
	}

	virtual void alloc() = 0;

	virtual void free() = 0;

	__device__ __host__
	virtual T* get_ptr() const
	{
		return ptr;
	}

	__device__ __host__
	virtual E get_extent() const
	{
		return extent;
	}

    virtual size_t get_size_in_bytes() const
    {
        return extent.get_number_of_elems() * sizeof(T);
    }
    
    T& operator[](const int idx) const {
        return ptr[idx];
    }

    T* begin() const
    {
        return ptr;
    }
    
    T* end() const
    {
        return ptr + extent.get_number_of_elems();
    }

	void incr_version()
	{
		++version;
	}

	int get_version() const
	{
		return version;
	}

	void set_version(const BaseBuffer<T, E>& buf)
	{
		version = buf.get_version();
	}

protected:

	const E extent;

	T* ptr;

	int version;

};

template <typename T, class E>
void show(BaseBuffer<T, E>& b)
{
	E e = b.get_extent();
	const T* ptr = b.get_ptr();
	for (int i = 0; i < e.get_number_of_elems(); i++)
	{
		std::cout << "b[" << i << "] = " << ptr[i] << std::endl;
	}
}

template <typename T, class E>
std::string show_as_vector(BaseBuffer<T, E>& b)
{
	std::stringstream ss;
	int i = 0;
	const int size = b.get_extent().get_number_of_elems();
	ss << "(";
	std::for_each(b.begin(), b.end(), [&i, size, &ss](float v)
	{
		ss << v;
		if (++i < size)
		{
			ss << ",";
		}
	});
	ss << ")";
	return ss.str();
}


