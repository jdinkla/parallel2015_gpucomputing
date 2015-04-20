/*
 * Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#pragma once

#include <map>
#include <functional>
#include <string>

class FunctionCaller
{
public:

	FunctionCaller(std::string _name);

	// add a function under key
	void add(std::string key, std::function<void()> function);

	// parse the command line arguments and execute a function
	int exec(int argc, char** argv);

private:

	void synopsis();

	std::string name;

	std::map<std::string, std::function<void()>> functions;

};