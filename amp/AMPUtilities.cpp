/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include <amp.h>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace concurrency;

void amp_info()
{
	vector<accelerator> accls = accelerator::get_all();

	if (accls.empty())
	{
		wcout << "No accelerators found that are compatible with C++ AMP" << std::endl << std::endl;
	}
	else
	{
		int i = 1;
		for (auto& acc : accls)
		{
			wcout << i << ". '" << acc.get_description() << "'" << endl
				<< "   is_emulated: " << acc.get_is_emulated() << endl
				<< "   version: " << acc.get_version() << endl
				<< endl;

			// TODO
			//std::wcout << "  " << i << ": " << a.description << " "
			//	<< std::endl << "       device_path                       = " << a.device_path
			//	<< std::endl << "       dedicated_memory                  = " << std::setprecision(4) << float(a.dedicated_memory) / (1024.0f * 1024.0f) << " Mb"
			//	<< std::endl << "       has_display                       = " << (a.has_display ? "true" : "false")
			//	<< std::endl << "       is_debug                          = " << (a.is_debug ? "true" : "false")
			//	<< std::endl << "       is_emulated                       = " << (a.is_emulated ? "true" : "false")
			//	<< std::endl << "       supports_double_precision         = " << (a.supports_double_precision ? "true" : "false")
			//	<< std::endl << "       supports_limited_double_precision = " << (a.supports_limited_double_precision ? "true" : "false")
			//	<< endl;

			i++;
		}
	}

}

vector<concurrency::accelerator> get_accs()
{
	vector<concurrency::accelerator> accls = accelerator::get_all();
	accls.erase(std::remove_if(accls.begin(), accls.end(), [](accelerator& a)
	{
		return a.is_emulated;
	}), accls.end());
	return accls;
}