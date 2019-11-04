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

/** @file FockStateSumRule.h
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/FockStateRule/SumRule.h"

#include <algorithm>

using namespace std;

namespace TBTK{
namespace FockStateRule{

SumRule::SumRule(
	std::initializer_list<Index> stateIndices,
	int numParticles
) :
	FockStateRule(FockStateRuleID::SumRule)
{
	for(unsigned int n = 0; n < stateIndices.size(); n++)
		this->stateIndices.push_back(*(stateIndices.begin()+n));

	sort(this->stateIndices.begin(), this->stateIndices.end());

	this->numParticles = numParticles;
}

SumRule::SumRule(
	std::vector<Index> stateIndices,
	int numParticles
) :
	FockStateRule(FockStateRuleID::SumRule)
{
	for(unsigned int n = 0; n < stateIndices.size(); n++)
		this->stateIndices.push_back(*(stateIndices.begin()+n));

	sort(this->stateIndices.begin(), this->stateIndices.end());

	this->numParticles = numParticles;
}

SumRule::~SumRule(
){
}

SumRule* SumRule::clone() const{
	return new SumRule(stateIndices, numParticles);
}

WrapperRule SumRule::createNewRule(
	const LadderOperator<BitRegister> &ladderOperator
) const{
	Index stateIndex = ladderOperator.getPhysicalIndex();
	LadderOperator<BitRegister>::Type type = ladderOperator.getType();

	int numParticles = this->numParticles;

	int sign = 0;
	if(type == LadderOperator<BitRegister>::Type::Creation)
		sign = 1;
	else
		sign = -1;

	for(unsigned int n = 0; n < stateIndices.size(); n++)
		if(stateIndices.at(n).equals(stateIndex, true))
			numParticles += sign;

	return WrapperRule(SumRule(stateIndices, numParticles));
}

WrapperRule SumRule::createNewRule(
	const LadderOperator<ExtensiveBitRegister> &ladderOperator
) const{
	Index stateIndex = ladderOperator.getPhysicalIndex();
	LadderOperator<ExtensiveBitRegister>::Type type = ladderOperator.getType();

	int numParticles = this->numParticles;

	int sign = 0;
	if(type == LadderOperator<ExtensiveBitRegister>::Type::Creation)
		sign = 1;
	else
		sign = -1;

	for(unsigned int n = 0; n < stateIndices.size(); n++)
		if(stateIndices.at(n).equals(stateIndex, true))
			numParticles += sign;

	return WrapperRule(SumRule(stateIndices, numParticles));
}

bool SumRule::isSatisfied(
	const FockSpace<BitRegister> &fockSpace,
	const FockState<BitRegister> &fockState
) const{
	int counter = 0;
	for(unsigned int n = 0; n < stateIndices.size(); n++)
		counter += fockSpace.getSumParticles(fockState, stateIndices.at(n));

	return (counter == numParticles);
}

bool SumRule::isSatisfied(
	const FockSpace<ExtensiveBitRegister> &fockSpace,
	const FockState<ExtensiveBitRegister> &fockState
) const{
	int counter = 0;
	for(unsigned int n = 0; n < stateIndices.size(); n++)
		counter += fockSpace.getSumParticles(fockState, stateIndices.at(n));

	return (counter == numParticles);
}

bool SumRule::operator==(const FockStateRule &rhs) const{
	switch(rhs.getFockStateRuleID()){
	case FockStateRuleID::WrapperRule:
		//The order is important. Infinite recursion will occur if rhs
		//is on the right.
		return (rhs == *this);
	case FockStateRuleID::SumRule:
		if(numParticles != ((const SumRule&)rhs).numParticles)
			return false;

		if(stateIndices.size() != ((const SumRule&)rhs).stateIndices.size())
			return false;

		for(unsigned int n = 0; n < stateIndices.size(); n++)
			if(!stateIndices.at(n).equals(((const SumRule&)rhs).stateIndices.at(n)))
				return false;

		return true;
	default:
		return false;
	}
}

void SumRule::print() const{
	Streams::out << "Sum(";
	for(unsigned int n = 0; n < stateIndices.size(); n++){
		if(n > 0)
			Streams::out << ", ";
		Streams::out << stateIndices.at(n).toString();
	}
	Streams::out << ") = " << numParticles;
}

};	//End of namespace FockStateRule
};	//End of namespace TBTK
