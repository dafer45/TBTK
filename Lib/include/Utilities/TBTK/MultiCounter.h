/* Copyright 2017 Kristofer Björnson
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
 *  @file Range.h
 *  @brief Helper class for flattening nested looping.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MULTI_COUNTER
#define COM_DAFER45_TBTK_MULTI_COUNTER

#include "TBTK/TBTKMacros.h"

#include <initializer_list>

namespace TBTK{

template<typename DataType>
class MultiCounter{
public:
	/** Constructor. */
	MultiCounter(
		const std::initializer_list<DataType> &begin,
		const std::initializer_list<DataType> &end,
		const std::initializer_list<DataType> &increment
	);

	/** Destructor. */
	~MultiCounter();

	/** Increment operator. */
	MultiCounter& operator++();

	/** Array subscript operator. */
	const DataType operator[](unsigned int n) const;

	/** Reset. */
	void reset();

	/** Returns true if the counter has reached the end. */
	bool done() const;
private:
	/** Values at which the iteration begins. */
	std::vector<DataType> begin;

	/** Values at which the iteration ends. */
	std::vector<DataType> end;

	/** Increments for the iteration. */
	std::vector<DataType> increment;

	/** Counters for the iteration. */
	std::vector<DataType> counter;
};

template<typename DataType>
inline MultiCounter<DataType>::MultiCounter(
	const std::initializer_list<DataType> &begin,
	const std::initializer_list<DataType> &end,
	const std::initializer_list<DataType> &increment
){
	TBTKAssert(
		begin.size() == end.size(),
		"MultiCounter::MultiCounter()",
		"'begin' and 'end' must have the same number of elements.",
		""
	);
	TBTKAssert(
		begin.size() == increment.size(),
		"MultiCounter::MultiCounter()",
		"'begin' and 'increment' must have the same number of"
		<< " elements.",
		""
	);

	this->begin = begin;
	this->end = end;
	this->increment = increment;

	for(unsigned int n = 0; n < begin.size(); n++){
		TBTKAssert(
			this->begin[n] <= this->end[n],
			"MultiCounter::MultiCounter()",
			"Only forward iteration is supported, but entry '" << n
			<< "' in 'end' is smaller than entry '" << n << "' in"
			<< " 'begin'.",
			""
		);

		TBTKAssert(
			this->increment[n] > 0,
			"MultiCounter::MultiCounter()",
			"Only positive increments are supported, but entry '"
			<< n << "' in 'increment' is '" << this->increment[n]
			<< "'.",
			""
		);
	}

	reset();
}

template<typename DataType>
inline MultiCounter<DataType>::~MultiCounter(){
}

template<typename DataType>
inline MultiCounter<DataType>& MultiCounter<DataType>::operator++(){
	for(int n = counter.size()-1; n > 0; n--){
		counter[n] += increment[n];
		if(counter[n] < end[n])
			return *this;

		counter[n] = begin[n];
	}

	counter[0] += increment[0];

	return *this;
}

template<typename DataType>
inline const DataType MultiCounter<DataType>::operator[](unsigned int n) const{
	return counter[n];
}

template<typename DataType>
inline void MultiCounter<DataType>::reset(){
	counter = begin;
}

template<typename DataType>
inline bool MultiCounter<DataType>::done() const{
	return counter[0] >= end[0];
}

}; //End of namesapce TBTK

#endif
