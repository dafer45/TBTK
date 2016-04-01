/** @package TBTKcalc
 *  @file Util.h
 *  @brief Utilities
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_UTIL
#define COM_DAFER45_TBTK_UTIL

#include <iostream>
#include <vector>
#include <chrono>

namespace TBTK{
namespace Util{

class Timer{
public:
	static void tick(std::string tag = "");
	static void tock();
private:
	static std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timestamps;
	static std::vector<std::string> tags;
};

inline void Timer::tick(std::string tag){
	timestamps.push_back(std::chrono::high_resolution_clock::now());
	tags.push_back(tag);
}

inline void Timer::tock(){
	std::chrono::time_point<std::chrono::high_resolution_clock> stop = std::chrono::high_resolution_clock::now();
	if(timestamps.size() > 0){
		std::chrono::time_point<std::chrono::high_resolution_clock> start = timestamps.back();
		timestamps.pop_back();
		std::string tag = tags.back();
		tags.pop_back();

		int hours = (std::chrono::duration_cast<std::chrono::hours>(stop - start).count());
		int minutes = (std::chrono::duration_cast<std::chrono::minutes>(stop - start).count())%60;
		int seconds = (std::chrono::duration_cast<std::chrono::seconds>(stop - start).count())%60;
		int milliseconds = (std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count())%1000;
		int microseconds = (std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count())%1000;
		int nanoseconds = (std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count())%1000;

		std::cout << "(" << timestamps.size() << ") ";
		if(hours > 0)
			std::cout << hours << "h ";
		if(hours > 0 || minutes > 0)
			std::cout << minutes << "m ";
		if(hours > 0 || minutes > 0 || seconds > 0)
			std::cout << seconds << "s ";
		if(hours > 0 || minutes > 0 || seconds > 0 || milliseconds > 0)
			std::cout << milliseconds << "ms ";
		if(hours > 0 || minutes > 0 || seconds > 0 || milliseconds > 0 || microseconds > 0)
			std::cout << microseconds << "us ";
		if(hours > 0 || minutes > 0 || seconds > 0 || milliseconds > 0 || microseconds > 0 || nanoseconds > 0)
			std::cout << nanoseconds << "ns ";
		std::cout << "\t" << tag << "\n";
	}
	else{
		std::cout << "Error in Time::tock(): No corresponding tick call made.\n";
	}
}

};
};

#endif
