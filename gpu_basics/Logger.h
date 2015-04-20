/*
 * Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.  
 *
 * See the LICENSE file in the root directory.
 */

#pragma once

#include <string>
#include <sstream>
//#include <memory>

/*
	This is a minimal logger. it has two levels DEBUG and INFO.
	If the code is compiled as DEBUG the log level is DEBUG, if it is compiled as RELEASE, 
	the log level is INFO.

	In log Level INFO the DEBUG messages are not shown. In log level DEBUG all messages are shown.

	for the implementation pattern see
	http://stackoverflow.com/questions/270947/can-any-one-provide-me-a-sample-of-singleton-in-c/271104#271104
*/

enum class LogLevel
{
	DEBUG, INFO
};

class Logger
{

public:

	static Logger& get_instance()
	{
		static Logger instance;
		return instance;
	}

	void debug(std::string msg) const;
	
	void info(std::string msg) const;

private:

	LogLevel level;

	// hidden constructor
	Logger();

	Logger(Logger const&) = delete;              // Don't Implement.

	void operator=(Logger const&) = delete;		 // Don't implement

};

#define DEBUG(msg) { std::stringstream str; str << msg; Logger::get_instance().debug(str.str()); }
#define INFO(msg) { std::stringstream str; str << msg; Logger::get_instance().info(str.str()); }

