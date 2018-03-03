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

/** @file ArrayState.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/ArrayState.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

ArrayState::ArrayState(
	initializer_list<unsigned int> resolution
) :
	AbstractState((AbstractState::StateID)1)
{
	storage = new Storage(resolution);
}

ArrayState::~ArrayState(){
	if(storage->release())
		delete storage;
}

ArrayState* ArrayState::clone() const{
	ArrayState* clonedState = new ArrayState(*this);
	storage->grab();

	return clonedState;
}

complex<double> ArrayState::getOverlap(const AbstractState &bra) const{
	TBTKNotYetImplemented("ArrayState::getOverlap()");
}

complex<double> ArrayState::getMatrixElement(
	const AbstractState &bra,
	const AbstractOperator &o
) const{
	TBTKNotYetImplemented("ArrayState::getOverlap()");
}

ArrayState::Storage::Storage(initializer_list<unsigned int> resolution){
	TBTKAssert(
		resolution.size() == 3,
		"ArrayState::Storage::Storage()",
		"Only three dimensional ArrayStates supported, but"
		<< " 'resolution' has " << resolution.size() << " components.",
		""
	);

	unsigned int size = 1;
	for(unsigned int n = 0; n < 3; n++){
		this->resolution.push_back(*(resolution.begin() + n));
		size *= this->resolution.at(n);
	}
	data = new complex<double>[size];

	referenceCounter = 1;
}

ArrayState::Storage::~Storage(){
	delete data;
}

};	//End of namespace TBTK
