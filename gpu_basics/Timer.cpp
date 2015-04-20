/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#define _CRT_SECURE_NO_WARNINGS

#include "Timer.h"
#include <sstream>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// see https://solarianprogrammer.com/2012/10/14/cpp-11-timing-code-performance/
void check_chrono()
{
	cout << "system_clock" << endl;
	cout << chrono::system_clock::period::num << endl;
	cout << chrono::system_clock::period::den << endl;
	cout << "steady = " << boolalpha << chrono::system_clock::is_steady << endl << endl;

	cout << "high_resolution_clock" << endl;
	cout << chrono::high_resolution_clock::period::num << endl;
	cout << chrono::high_resolution_clock::period::den << endl;
	cout << "steady = " << boolalpha << chrono::high_resolution_clock::is_steady << endl << endl;

	cout << "steady_clock" << endl;
	cout << chrono::steady_clock::period::num << endl;
	cout << chrono::steady_clock::period::den << endl;
	cout << "steady = " << boolalpha << chrono::steady_clock::is_steady << endl << endl;
}

string format_time(std::chrono::system_clock::time_point tp)
{
	time_t ttp = system_clock::to_time_t(tp);
	tm* tm = gmtime(&ttp);
	const milliseconds allMs = duration_cast<milliseconds>(tp.time_since_epoch());
	const size_t ms = allMs.count() % 1000;
	stringstream ss;
	ss
		<< 1900 + tm->tm_year
		<< "-"
		<< setfill('0') << setw(2)
		<< 1 + tm->tm_mon
		<< "-"
		<< setfill('0') << setw(2)
		<< tm->tm_mday
		<< " "
		<< setfill('0') << setw(2)
		<< tm->tm_hour
		<< ":"
		<< setfill('0') << setw(2)
		<< tm->tm_min
		<< ":"
		<< setfill('0') << setw(2)
		<< tm->tm_sec
		<< "."
		<< setfill('0') << setw(3)
		<< ms;
	return ss.str();
}

string format_time()
{
	return format_time(system_clock::now());
}



