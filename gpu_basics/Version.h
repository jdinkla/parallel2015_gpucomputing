/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

class Version
{

public:

	void incr_version()
	{
		++version;
	}

	int get_version() const
	{
		return version;
	}

	void set_version(const int version)
	{
		this->version = version;
	}

protected:

	int version;

};
