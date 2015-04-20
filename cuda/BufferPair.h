/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "PinnedBuffer.h"
#include "DeviceBuffer.h"

template <typename T, class E = Extent2>
class BufferPair
{

public:

	BufferPair(PinnedBuffer<T, E>& _host, DeviceBuffer<T, E>& _device);

	PinnedBuffer<T, E>& get_host() const
	{
		return host;
	}

	DeviceBuffer<T, E>& get_device() const
	{
		return device;
	}

	void update_host();

	void update_device();

private:

	PinnedBuffer<T, E>& host;

	DeviceBuffer<T, E>& device;

};

/*
template <typename T, class E>
void copy_h2d(PinnedBuffer<T, E>& host, DeviceBuffer<T, E>& device)
{
	if (host.get_version() > device.get_version())
	{
		device.copy_from(host);
	}
}

template <typename T, class E>
void copy_d2h(PinnedBuffer<T, E>& host, DeviceBuffer<T, E>& device);
*/
