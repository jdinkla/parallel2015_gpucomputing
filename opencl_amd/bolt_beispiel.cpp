/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "bolt_beispiel.h"

#include <iostream>
#include <bolt/cl/transform.h>
#include <bolt/amp/transform.h>
#include <vector>

using namespace std;

BOLT_FUNCTOR(Functor,
struct Functor {
	int operator() (const int v) const
	{
		return v*v;
	};
};);
void bolt_beispiel()
{
	std::vector<int> v(16);
	std::iota(v.begin(), v.end(), 1);
	bolt::cl::transform(v.begin(), v.end(), v.begin(), Functor{});
	bolt::amp::transform(v.begin(), v.end(), v.begin(), 
		[=](const int v) restrict(cpu, amp)	{
			return v*v;
	});
	for (int i = 0; i < v.size(); i++) {
		cout << "v[" << i << "] = " << v[i] << endl;
	}
}



