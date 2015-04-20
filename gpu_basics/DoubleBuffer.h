/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

template <class Buffer>
class DoubleBuffer
{
public:

	DoubleBuffer(Buffer _a, Buffer _b)
		: a(_a), b(_b)
	{
		value = 0;
	}

	Buffer get_current()
	{
		return value ? b : a;
	}
	
	Buffer get_previous()
	{
		return value ? a : b;
	}

	void swap()
	{
		value = !value;
	}

	void set_a(Buffer _a)
	{
		a = _a;
	}

	void set_b(Buffer _b)
	{
		b = _b;
	}

	int get_value() const
	{
		return value;
	}

	Buffer get(const int i) const
	{
		if (i == 0)
		{
			return a;
		}
		else if (i == 1)
		{
			return b;
		}
		else
		{
			throw std::runtime_error("DoubleBuffer<T> wrong index");
		}
	}

	Buffer operator[](const int i) const
	{
		return get(i);
	}

private:
	Buffer a;
	Buffer b;
	int value;
};



