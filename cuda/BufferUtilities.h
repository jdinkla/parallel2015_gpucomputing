/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <iostream>
#include <vector>
#include "ManagedBuffer.h"
#include "PinnedBuffer.h"
#include "DeviceBuffer.h"

template <typename T, class E>
using pinned_buffers = std::vector < std::shared_ptr<PinnedBuffer<T, E>> > ;

template <typename T, class E>
using device_buffers = std::vector < std::shared_ptr<DeviceBuffer<T, E>> >;


//template <typename T, class E>
//void show(ManagedBuffer<T, E>& m)
//{
//	E e = m.get_extent();
//	const T* d = m.get_ptr();
//	for (int i = 0; i < e.get_number_of_elems(); i++)
//	{
//		std::cout << "d[" << i << "] = " << d[i] << std::endl;
//	}
//}
//
//template <typename T, class E>
//void show(PinnedBuffer<T, E>& m)
//{
//	E e = m.get_extent();
//	const T* d = m.get_ptr();
//	for (int i = 0; i < e.get_number_of_elems(); i++)
//	{
//		std::cout << "d[" << i << "] = " << d[i] << std::endl;
//	}
//}

template <typename T, class E>
void sequence(HostBuffer<T, E>& b, const int start = 0)
{
	E e = b.get_extent();
	T* ptr = b.get_ptr();
	for (int i = 0; i < e.get_number_of_elems(); i++)
	{
		ptr[i] = (T) (start + i + 1);
	}
}

template <typename T, class E>
double checksum(HostBuffer<T, E>& b)
{
	E e = b.get_extent();
	T* ptr = b.get_ptr();
	double sum = 0.0;
	for (int i = 0; i < e.get_number_of_elems(); i++)
	{
		sum += (double)ptr[i];
	}
	return sum;
}

template <typename T, class E>
pinned_buffers<T, E> create_pinned_buffers(const int numBuffers, const Extent2& extent)
{
	pinned_buffers<T, E> buffers;
	for (int i = 0; i < numBuffers; i++)
	{
		PinnedBuffer<T, E>* ptr = new PinnedBuffer<T, E>(extent);
		std::shared_ptr<PinnedBuffer<T, E>> sptr(ptr);
		buffers.push_back(sptr);
		sptr->alloc();
		sequence(*sptr, i);
	}
	return buffers;
}
