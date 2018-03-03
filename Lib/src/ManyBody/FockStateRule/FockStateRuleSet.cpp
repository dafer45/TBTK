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

/** @file FockStateRuleSet.h
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/FockStateRuleSet.h"

#include <algorithm>
#include <vector>

using namespace std;

namespace TBTK{

FockStateRuleSet::FockStateRuleSet(){
}

FockStateRuleSet::~FockStateRuleSet(){
}

bool FockStateRuleSet::isSatisfied(
	const FockSpace<BitRegister> &fockSpace,
	const FockState<BitRegister> &fockState
) const{
	bool isSatisfied = true;
	for(unsigned int n = 0; n < fockStateRules.size(); n++){
		if(!fockStateRules.at(n).isSatisfied(fockSpace, fockState)){
			isSatisfied = false;
			break;
		}
	}

	return isSatisfied;
}

bool FockStateRuleSet::isSatisfied(
	const FockSpace<ExtensiveBitRegister> &fockSpace,
	const FockState<ExtensiveBitRegister> &fockState
) const{
	bool isSatisfied = true;
	for(unsigned int n = 0; n < fockStateRules.size(); n++){
		if(!fockStateRules.at(n).isSatisfied(fockSpace, fockState)){
			isSatisfied = false;
			break;
		}
	}

	return isSatisfied;
}

bool FockStateRuleSet::operator==(const FockStateRuleSet &rhs) const{
	if(fockStateRules.size() != rhs.fockStateRules.size())
		return false;

	vector<int> permutation;
	for(unsigned int n = 0; n < fockStateRules.size(); n++)
		permutation.push_back(n);

	do{
		bool ruleSetsAreEqual = true;
		for(unsigned int n = 0; n < fockStateRules.size(); n++){
			if(!(fockStateRules.at(permutation.at(n)) == rhs.fockStateRules.at(n))){
				ruleSetsAreEqual = false;
				break;
			}
		}

		if(ruleSetsAreEqual)
			return true;
	}while(next_permutation(permutation.begin(), permutation.end()));

	return false;
}

};	//End of namespace TBTK
