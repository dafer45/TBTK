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

/** @file PropertyExtractor.cpp
 *
 *  @author Kristofer Björnson
 */

#include "PropertyExtractor.h"
#include "Streams.h"

using namespace std;

namespace TBTK{

namespace{
	complex<double> i(0,1);
}

PropertyExtractor::PropertyExtractor(){
}

PropertyExtractor::~PropertyExtractor(){
}

void PropertyExtractor::calculate(
	void (*callback)(
		PropertyExtractor *cb_this,
		void *memory,
		const Index &index,
		int offset
	),
	void *memory,
	Index pattern,
	const Index &ranges,
	int currentOffset,
	int offsetMultiplier
){
	int currentSubindex = pattern.size()-1;
	for(; currentSubindex >= 0; currentSubindex--){
		if(pattern.at(currentSubindex) < 0)
			break;
	}

	if(currentSubindex == -1){
		callback(this, memory, pattern, currentOffset);
	}
	else{
		int nextOffsetMultiplier = offsetMultiplier;
		if(pattern.at(currentSubindex) < IDX_SUM_ALL)
			nextOffsetMultiplier *= ranges.at(currentSubindex);
		bool isSumIndex = false;
		if(pattern.at(currentSubindex) == IDX_SUM_ALL)
			isSumIndex = true;
		for(int n = 0; n < ranges.at(currentSubindex); n++){
			pattern.at(currentSubindex) = n;
			calculate(callback,
					memory,
					pattern,
					ranges,
					currentOffset,
					nextOffsetMultiplier
			);
			if(!isSumIndex)
				currentOffset += offsetMultiplier;
		}
	}
}

void PropertyExtractor::ensureCompliantRanges(
	const Index &pattern,
	Index &ranges
){
	for(unsigned int n = 0; n < pattern.size(); n++){
		if(pattern.at(n) >= 0)
			ranges.at(n) = 1;
	}
}

void PropertyExtractor::getLoopRanges(
	const Index &pattern,
	const Index &ranges,
	int *lDimensions,
	int **lRanges
){
	*lDimensions = 0;
	for(unsigned int n = 0; n < ranges.size(); n++){
		if(pattern.at(n) < IDX_SUM_ALL)
			(*lDimensions)++;
	}

	(*lRanges) = new int[*lDimensions];
	int counter = 0;
	for(unsigned int n = 0; n < ranges.size(); n++){
		if(pattern.at(n) < IDX_SUM_ALL)
			(*lRanges)[counter++] = ranges.at(n);
	}
}

};	//End of namespace TBTK
