/* Copyright 2019 Kristofer Björnson
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

/** @file BasisStateSet.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/BasisStateSet.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

using namespace std;
//using namespace nlohmann;

namespace TBTK{

BasisStateSet::BasisStateSet(){
}

BasisStateSet::BasisStateSet(
	const string &serialization,
	Mode mode
){
	TBTKNotYetImplemented("BasisStateSet::BasisStateSet()");
}

BasisStateSet::~BasisStateSet(){
	for(
		IndexedDataTree<AbstractState*>::Iterator iterator
			= basisStateTree.begin();
		iterator != basisStateTree.end();
		++iterator
	){
		delete *iterator;
	}
}

BasisStateSet::Iterator BasisStateSet::begin(){
	return BasisStateSet::Iterator(basisStateTree);
}

BasisStateSet::ConstIterator BasisStateSet::begin() const{
	return BasisStateSet::ConstIterator(basisStateTree);
}

BasisStateSet::ConstIterator BasisStateSet::cbegin() const{
	return BasisStateSet::ConstIterator(basisStateTree);
}

BasisStateSet::Iterator BasisStateSet::end(){
	return BasisStateSet::Iterator(basisStateTree, true);
}

BasisStateSet::ConstIterator BasisStateSet::end() const{
	return BasisStateSet::ConstIterator(basisStateTree, true);
}

BasisStateSet::ConstIterator BasisStateSet::cend() const{
	return BasisStateSet::ConstIterator(basisStateTree, true);
}

string BasisStateSet::serialize(Mode mode) const{
	TBTKNotYetImplemented("BasisStateSet::serialize()");
}

};	//End of namespace TBTK
