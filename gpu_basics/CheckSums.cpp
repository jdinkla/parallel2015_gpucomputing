/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "CheckSums.h"
#include <iostream>

using namespace std;

template <typename T>
CheckSums<T>::CheckSums()
{
}

template <typename T>
double CheckSums<T>::checksum(const T* ptr, const int number_of_elems)
{
	double local_sum = 0.0;
	for (int i = 0; i < number_of_elems; i++)
	{
		local_sum += (double)ptr[i];
	}
	return local_sum;
}

template <typename T>
void CheckSums<T>::add(const double _sum)
{
	sum += _sum;
	checksums.push_back(_sum);
}

template <typename T>
void CheckSums<T>::add(const T* ptr, const int number_of_elems)
{
	double local_sum = checksum(ptr, number_of_elems);
	sum += local_sum;
	checksums.push_back(sum);
}

template <typename T>
void CheckSums<T>::report() const
{
	int i = 0;
	if (checksums.size() == 0)
	{
		cout << "no check sum to report" << endl;
	}
	else if (checksums.size() == 1)
	{
		cout << "checksum = " << sum << endl;
	}
	else
	{
		// 2 or more
		int i = 0;
		for (auto s : checksums)
		{
			cout << "check sum[" << i++ << "]=" << s << endl;
		}
		cout << "check sum=" << sum << endl;
	}
}

// Instances

template class CheckSums<int>;
template class CheckSums<float>;

