/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <vector>

template <typename T>
class CheckSums
{
public:

	CheckSums();
	
	// calculate a checksum
	double checksum(const T* ptr, const int number_of_elems);

	// calculate a checksum and add it
	void add(const T* ptr, const int number_of_elems);

	// add a check sum
	void add(const double _sum);
	
	double get_sum() const
	{
		return sum;
	}

	// print a report
	void report() const;

private:

	double sum = 0.0;

	std::vector<double> checksums;

};