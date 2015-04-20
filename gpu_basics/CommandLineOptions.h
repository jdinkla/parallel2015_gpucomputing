/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

// see http://stackoverflow.com/questions/865668/parse-command-line-arguments/868894#868894

#include <algorithm>
#include <string>

inline
char* get_cmd_option(char** begin, char** end, const std::string option)
{
	char ** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return 0;
}

inline
bool exists_cmd_option(char** begin, char** end, const std::string option)
{
	return std::find(begin, end, option) != end;
}


