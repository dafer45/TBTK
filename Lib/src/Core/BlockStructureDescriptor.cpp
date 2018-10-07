/* Copyright 2018 Kristofer Björnson
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

/** @file BlockStructureDescriptor.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/BlockStructureDescriptor.h"

using namespace std;

namespace TBTK{

BlockStructureDescriptor::BlockStructureDescriptor(){
}

BlockStructureDescriptor::BlockStructureDescriptor(
	const HoppingAmplitudeSet &hoppingAmplitudeSet
){
	IndexTree blockIndices = hoppingAmplitudeSet.getSubspaceIndices();
	for(
		IndexTree::ConstIterator blockIterator = blockIndices.cbegin();
		blockIterator != blockIndices.cend();
		++blockIterator
	){
		HoppingAmplitudeSet::ConstIterator iterator
			= hoppingAmplitudeSet.cbegin(*blockIterator);
		numStatesInBlock.push_back(iterator.getNumBasisIndices());
	}

	if(numStatesInBlock.size() == 0){
		numStatesInBlock.push_back(hoppingAmplitudeSet.getBasisSize());
		blockToStateMap.push_back(0);
		for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++)
			stateToBlockMap.push_back(0);
	}
	else{
		unsigned int blockCounter = 0;
		unsigned int intraBlockCounter = 0;
		for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++){
			if(intraBlockCounter >= numStatesInBlock.at(blockCounter)){
				intraBlockCounter = 0;
				blockCounter++;
			}

			if(intraBlockCounter == 0)
				blockToStateMap.push_back(n);

			stateToBlockMap.push_back(blockCounter);
			intraBlockCounter++;
		}
	}
}

};	//End of namespace TBTK
