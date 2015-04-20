/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

// very simple Maybe/Optional type

template <class T>
class Maybe
{
public:

	Maybe()
		: isJust(false)
	{
	}

	Maybe(T _value)
		: isJust(true)
		, value(_value)
	{
	}

	bool is_just()
	{
		return isJust;
	}

	T get_value()
	{
		return value;
	}

private:

	bool isJust = false;

	T value;

};