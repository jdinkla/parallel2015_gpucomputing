#include "Logger.h"
#include <chrono>
#include <iostream>
#include "Timer.h"

using namespace std;

void message(std::string msg)
{
	cout << format_time() << " " << msg << endl;
}

Logger::Logger()
{
#ifdef _DEBUG
	level = LogLevel::DEBUG;
#else
	level = LogLevel::INFO;
#endif
}

void Logger::debug(std::string msg) const
{
	if (level == LogLevel::DEBUG)
	{
		message(msg);
	}
}

void Logger::info(std::string msg) const
{
	message(msg);
}
