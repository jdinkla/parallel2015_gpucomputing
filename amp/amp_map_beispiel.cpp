/*
 * Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#include <amp.h>
#include <iostream>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace concurrency;

void amp_map_beispiel()
{
	std::vector<int> v(16);
	std::iota(v.begin(), v.end(), 1);
	array_view<int, 1> av(16, v);
	parallel_for_each(av.extent, [=](index<1> i) restrict(amp) 
	{
		const int v = av[i];
		av[i] = v * v;
	});
	av.synchronize();
	for (int i = 0; i < v.size(); i++) {
		cout << "v[" << i << "] = " << v[i] << endl;
	}
}

void amp_map_beispiel2()
{
	std::vector<int> v(16);
	std::iota(v.begin(), v.end(), 1);
	array_view<int, 1> av(16, v);
	parallel_for_each(av.extent, [=](index<1> i) restrict(amp)
	{
		av[i] = av[i] * av[i];
	});
	av.synchronize();

	cout << "v=("; string sep = "";
	std::for_each(v.begin(), v.end(), [&sep](int v)
	{
		cout << sep << v; sep = ",";
	});
	cout << ")" << endl;
}

void amp_map_beispiel3()
{
	std::vector<int> v(16);
	std::iota(v.begin(), v.end(), 1);

	array_view<int, 1> av(16, v);
	accelerator acc = accelerator::get_all()[1];    // Benötigt 2 GPUs!
	accelerator_view view = acc.get_default_view();
	parallel_for_each(view, av.extent, [=](index<1> i) restrict(amp)
	{
		const int v = av[i];
		av[i] = v * v;
	});
	av.synchronize();

	for (int i = 0; i < v.size(); i++)
	{
		cout << "v[" << i << "] = " << v[i] << endl;
	}
}

void amp_map_beispiel4()
{
	accelerator acc = accelerator::get_all()[1];
	accelerator_view view = acc.get_default_view();
	array<float, 1> a(16, view);

}

void test_d2d_copy()
{
	vector<accelerator> accs = accelerator::get_all();
	array<float, 1> src(16, accs[0].get_default_view());
	array<float, 1> dest(16, accs[1].get_default_view());
	completion_future f = copy_async(src, dest);
	f.get();
}

