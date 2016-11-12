/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @package TBTKcalc
 *  @file Util.h
 *  @brief Utilities
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_UTIL
#define COM_DAFER45_TBTK_UTIL

#include "Streams.h"

#include <vector>
#include <chrono>

namespace TBTK{
namespace Util{

/** A Timer for measuring execution times. A sequence of tick-tock calls will
 *  meassure and print the time from tick to tock. Multiple timestamps can be
 *  pushed onto the stack by repeated tick calls, and subsequent tock calls
 *  will return the time between the most recent tick call and then pop it.
 */
class Timer{
public:
	/** Bush timestamp onto stack.
	 *  @param tag Optional identifier tag that will be printed together
	 *  with the elapsed time at subsequent tock call. */
	static void tick(std::string tag = "");

	/** Pop timestamp from stack and print elapsed time and identifier
	 *  tag. */
	static void tock();
private:
	/** Timestamp stack. */
	static std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timestamps;

	/** Tag stack. */
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

		Streams::out << "(" << timestamps.size() << ") ";
		if(hours > 0)
			Streams::out << hours << "h ";
		if(hours > 0 || minutes > 0)
			Streams::out << minutes << "m ";
		if(hours > 0 || minutes > 0 || seconds > 0)
			Streams::out << seconds << "s ";
		if(hours > 0 || minutes > 0 || seconds > 0 || milliseconds > 0)
			Streams::out << milliseconds << "ms ";
		if(hours > 0 || minutes > 0 || seconds > 0 || milliseconds > 0 || microseconds > 0)
			Streams::out << microseconds << "us ";
		if(hours > 0 || minutes > 0 || seconds > 0 || milliseconds > 0 || microseconds > 0 || nanoseconds > 0)
			Streams::out << nanoseconds << "ns ";
		Streams::out << "\t" << tag << "\n";
	}
	else{
		Streams::out << "Error in Time::tock(): No corresponding tick call made.\n";
	}
}

};	//End of namespace Util
};	//End of namespace TBTK

#endif
