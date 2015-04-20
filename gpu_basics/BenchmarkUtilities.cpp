/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "BenchmarkUtilities.h"

using namespace std;

vector<int> get_sizes()
{
	vector<int> result;
	for (int i = 1; i <= 16; i++)
	{
		result.push_back(i * 1024);
	}
	return result;
}
