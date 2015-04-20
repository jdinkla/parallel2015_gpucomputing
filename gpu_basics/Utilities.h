/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <iostream>
#include <cmath>

// print_csv("x", "j", "j", "variable number of args")
// print_csv(1, 2, 3, 4)
template <typename T>
void print_csv(const T& str)
{
	std::cout << str << std::endl;
}

template <typename T, typename... Args>
void print_csv(const T& str, Args... args)
{
	std::cout << str << ";";
	print_csv(args...);
}

template <class T>
T align(const T size, const T align)
{
	const long long av = align - 1;
	const long long val = ((long long)size) & ~av;
	return (T)val;
}

// Ganzzahlige Division, die immer nach oben rundet
// ceiling(x/y), z. B. ceiling_div(5, 2) = (5 + 1) / 2 = 3
inline int ceiling_div(const int x, const int y)
{
	return (x + y - 1) / y;
}

// nmo(128, 127) = 128, nmo(128, 128) = 128, nmo(129) = 256
inline int next_multiple_of(const int m, const int v)
{
	return (int) (floor((v + m - 1) / m) * m);
}


