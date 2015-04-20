/*
 * Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#include "FunctionCaller.h"
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

// TODO suffixe werden eingefügt, weil wahrscheinlich nicht contains aufgerufen wird
// test mit functions[dhdhdhdhdkk] fügt hinzu

FunctionCaller::FunctionCaller(std::string _name)
	: name(_name)
{
	auto syn = [this]()
	{
		this->synopsis();
	};
	add("help", syn);
	add("?", syn);
}

void FunctionCaller::add(std::string key, std::function<void()> function)
{
	transform(key.begin(), key.end(), key.begin(), ::tolower);
	functions[key] = function;
}

int FunctionCaller::exec(int argc, char** argv)
{
	int rc = 0;
	if (argc != 2)
	{
		cerr << "ERROR: wrong number of arguments!" << endl;
		functions["help"]();
		return 1;
	}
	else
	{
		string key = argv[1];
		std::transform(key.begin(), key.end(), key.begin(), ::tolower);
		auto f = functions[key];
		if (f)
		{
			try
			{
				f();
			}
			catch (exception& e)
			{
				cerr << "ERROR: " << e.what() << endl;
				rc = 2;
			}
		} 
		else  
		{
			cerr << "ERROR: unknown argument!" << endl;
			functions["help"]();
			return 1;
		}
	}
	return rc;
}

void FunctionCaller::synopsis()
{
	cout << "SYNOPSIS: " << name << " ARG" << endl;
	cout << "  where ARG is one of { ";
	vector<string> keys;
	int elemsToGo = (int) functions.size();
	for (auto f : functions)
	{
		cout << f.first;
		if (elemsToGo-- > 1)
		{
			cout << ", ";
		}
	}
	cout << " }" << endl;
}
