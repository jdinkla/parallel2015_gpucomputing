/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

class ITimer
{
public:

	virtual void start() = 0;

	virtual void stop() = 0;

	virtual float delta() = 0;

};
