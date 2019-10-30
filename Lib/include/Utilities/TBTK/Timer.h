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
 *  @file Timer.h
 *  @brief A Timer for measuring execution time.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_TIMER
#define COM_DAFER45_TBTK_TIMER

#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <chrono>
#include <iomanip>
#include <vector>

namespace TBTK{

/** @brief A Timer for measuring execution time.
 *
 *  # Modes
 *  The Timer has two different modes of execution and it is possible to use
 *  these two modes at the same time.
 *
 *  ## Timestamp stack
 *  A pair of tick-tock calls (with an optional tag as argument to the tick
 *  call) can be used to measure the time taken to execute a code segment.
 *  ```cpp
 *    Timer::tick("MyTag");
 *    //Code
 *    //...
 *    Timer::tock();
 *  ```
 *  The tick call pushes a timestamp and tag onto a stack, which is poped by
 *  the tock call. It is therefore possible to nest tick-tock calls to
 *  simultaneously measure the execution time for a large block of code and its
 *  smaller sections. When the timestamp is poped, the time since the tick call
 *  and the corresponding tag is printed to Streams::out.
 *
 *  ## Accumulators
 *  An accumulator can be created using
 *  ```cpp
 *    unsigned int id = Timer::creteAccumulator("AccumulatorTag");
 *  ```
 *  If the ID is passed to the tick and tock calls, the time between the calls
 *  is added to a corresponding accumulator. This can be used to measure the
 *  total time required to execute a specific code segment that for example is
 *  executed inside a loop.
 *
 *  To print the time accumulated in the accumulators, we use
 *  ```cpp
 *    Timer::printAccumulators();
 *  ```
 *
 *  # Example
 *  \snippet Utilities/Timer.cpp Timer
 *  ## Output
 *  \snippet output/Utilities/Timer.output Timer */
class Timer{
public:
	/** Push timestamp onto stack.
	 *
	 *  @param tag Optional identifier tag that will be printed together
	 *  with the elapsed time at subsequent tock call. */
	static void tick(std::string tag = "");

	/** Pop timestamp from stack and print elapsed time and identifier
	 *  tag. */
	static void tock();

	/** Create an accumulator that can be used to accumulate multiple time
	 *  measurements.
	 *
	 *  @param tag Optional identifier tag that will be printed together
	 *  with the accumulated time.
	 *
	 *  @return An id that can be passed to tick-tock calls to use the
	 *  accumulator. */
	static unsigned int createAccumulator(const std::string &tag = "");

	/** Initiate a time interval to be added to an accumulator.
	 *
	 *  @param id The ID of the accumulator to use. */
	static void tick(unsigned int id);

	/** Finalize a time interval and add it to an accumulator.
	 *
	 *  @param id The ID of the accumulator to stop. */
	static void tock(unsigned int id);

	/** Reset accumulator.
	 *
	 *  @param id The ID of the accumulator to reset. */
	static void resetAccumulator(unsigned int id);

	/** Reset all accumulators. */
	static void resetAccumulators();

	/** Print accumulators. */
	static void printAccumulators();
private:
	/** Timestamp stack. */
	static std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timestamps;

	/** Tag stack. */
	static std::vector<std::string> tags;

	/** Accumulator timestamps. */
	static std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> accumulatorTimestamps;

	/** Accumulator tags. */
	static std::vector<std::string> accumulatorTags;

	/** Accumulators. */
	static std::vector<long> accumulators;
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

inline unsigned int Timer::createAccumulator(const std::string &tag){
	accumulatorTimestamps.push_back(
		std::chrono::high_resolution_clock::now()
	);
	accumulatorTags.push_back(tag);
	accumulators.push_back(0);

	return accumulators.size() - 1;
}

inline void Timer::tick(unsigned int id){
	TBTKAssert(
		id < accumulators.size(),
		"Timer::tick()",
		"'id' is out of bounds.",
		"Ensure that the id corresponds to a value returned by a"
		<< " corresponding call to Timer::createAccumulator()."
	);
	accumulatorTimestamps[id] = std::chrono::high_resolution_clock::now();
}

inline void Timer::tock(unsigned int id){
	TBTKAssert(
		id < accumulators.size(),
		"Timer::tock()",
		"'id' is out of bounds.",
		"Ensure that the id corresponds to a value returned by a"
		<< " corresponding call to Timer::createAccumulator()."
	);
	std::chrono::time_point<std::chrono::high_resolution_clock> stop
		= std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> start
		= accumulatorTimestamps[id];

	accumulators[id]
		+= std::chrono::duration_cast<std::chrono::nanoseconds>(
			stop - start
		).count();
}

inline void Timer::resetAccumulator(unsigned int id){
	TBTKAssert(
		id < accumulators.size(),
		"Timer::resetAccumulator()",
		"'id' is out of bounds.",
		"Ensure that the id corresponds to a value returned by a"
		<< " corresponding call to Timer::createAccumulator()."
	);

	accumulators[id] = 0;
}

inline void Timer::resetAccumulators(){
	for(unsigned int n = 0; n < accumulators.size(); n++)
		accumulators[n] = 0;
}

inline void Timer::printAccumulators(){
	Streams::out << "============================== Accumulator table ==============================\n";
	Streams::out << std::left << std::setw(10) << "ID" << std::setw(33) << "Time" << std::setw(100) << "     Tag" << "\n";
	for(unsigned int n = 0; n < accumulators.size(); n++){
		long time = accumulators[n];

		long hours = time/(60ll*60ll*1000ll*1000ll*1000ll);
		long minutes = (time/(60ll*1000ll*1000ll*1000ll))%60ll;
		long seconds = (time/(1000ll*1000ll*1000ll))%60ll;
		long milliseconds = (time/(1000ll*1000ll))%1000ll;
		long microseconds = (time/(1000ll))%1000ll;
		long nanoseconds = time%1000ll;

		const std::string &tag = accumulatorTags[n];

		Streams::out << std::left << std::setw(10) << "[" + std::to_string(n) + "]" << std::right;
		if(hours > 0)
			Streams::out << std::setw(6) << std::to_string(hours) + "h";
		else
			Streams::out << std::setw(6) << " ";
		if(hours > 0 || minutes > 0)
			Streams::out << std::setw(5) << std::to_string(minutes) + "m";
		else
			Streams::out << std::setw(5) << " ";
		if(hours > 0 || minutes > 0 || seconds > 0)
			Streams::out << std::setw(4) << std::to_string(seconds) + "s";
		else
			Streams::out << std::setw(4) << " ";
		if(hours > 0 || minutes > 0 || seconds > 0 || milliseconds > 0)
			Streams::out << std::setw(6) << std::to_string(milliseconds) + "ms";
		else
			Streams::out << std::setw(6) << " ";
		if(hours > 0 || minutes > 0 || seconds > 0 || milliseconds > 0 || microseconds > 0)
			Streams::out << std::setw(6) << std::to_string(microseconds) + "us";
		else
			Streams::out << std::setw(6) << " ";
		if(hours > 0 || minutes > 0 || seconds > 0 || milliseconds > 0 || microseconds > 0 || nanoseconds > 0)
			Streams::out << std::setw(6) << std::to_string(nanoseconds) + "ns";
		else
			Streams::out << std::setw(6) << " ";
		Streams::out << std::left << "     " << std::setw(100) << tag << "\n";
	}
	Streams::out << "===============================================================================\n";
}

};	//End of namespace TBTK

#endif
