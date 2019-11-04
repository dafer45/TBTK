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

/** @file DifferenceRule.h
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/FockStateRule/DifferenceRule.h"
#include "TBTK/FockSpace.h"

#include <algorithm>

using namespace std;

namespace TBTK{
namespace FockStateRule{

DifferenceRule::DifferenceRule(
	std::initializer_list<Index> addStateIndices,
	std::initializer_list<Index> subtractStateIndices,
	int difference
) :
	FockStateRule(FockStateRuleID::DifferenceRule)
{
	for(unsigned int n = 0; n < addStateIndices.size(); n++)
		this->addStateIndices.push_back(*(addStateIndices.begin()+n));

	sort(this->addStateIndices.begin(), this->addStateIndices.end());

	for(unsigned int n = 0; n < subtractStateIndices.size(); n++)
		this->subtractStateIndices.push_back(*(subtractStateIndices.begin()+n));

	sort(
		this->subtractStateIndices.begin(),
		this->subtractStateIndices.end()
	);

	this->difference = difference;
}

DifferenceRule::DifferenceRule(
	std::vector<Index> addStateIndices,
	std::vector<Index> subtractStateIndices,
	int difference
) :
	FockStateRule(FockStateRuleID::DifferenceRule)
{
	for(unsigned int n = 0; n < addStateIndices.size(); n++)
		this->addStateIndices.push_back(*(addStateIndices.begin()+n));

	sort(this->addStateIndices.begin(), this->addStateIndices.end());

	for(unsigned int n = 0; n < subtractStateIndices.size(); n++)
		this->subtractStateIndices.push_back(*(subtractStateIndices.begin()+n));

	sort(
		this->subtractStateIndices.begin(),
		this->subtractStateIndices.end()
	);

	this->difference = difference;
}

DifferenceRule::~DifferenceRule(
){
}

DifferenceRule* DifferenceRule::clone() const{
	return new DifferenceRule(
		addStateIndices,
		subtractStateIndices,
		difference
	);
}

WrapperRule DifferenceRule::createNewRule(
	const LadderOperator<BitRegister> &ladderOperator
) const{
	Index stateIndex = ladderOperator.getPhysicalIndex();
	LadderOperator<BitRegister>::Type type = ladderOperator.getType();

	int difference = this->difference;

	int sign;
	if(type == LadderOperator<BitRegister>::Type::Creation)
		sign = 1;
	else
		sign = -1;

	for(unsigned int n = 0; n < addStateIndices.size(); n++)
		if(addStateIndices.at(n).equals(stateIndex, true))
			difference += sign;

	for(unsigned int n = 0; n < subtractStateIndices.size(); n++)
		if(subtractStateIndices.at(n).equals(stateIndex, true))
			difference -= sign;

	return WrapperRule(
		DifferenceRule(
			addStateIndices,
			subtractStateIndices,
			difference
		)
	);
}

WrapperRule DifferenceRule::createNewRule(
	const LadderOperator<ExtensiveBitRegister> &ladderOperator
) const{
	Index stateIndex = ladderOperator.getPhysicalIndex();
	LadderOperator<ExtensiveBitRegister>::Type type = ladderOperator.getType();

	int difference = this->difference;

	int sign;
	if(type == LadderOperator<ExtensiveBitRegister>::Type::Creation)
		sign = 1;
	else
		sign = -1;

	for(unsigned int n = 0; n < addStateIndices.size(); n++)
		if(addStateIndices.at(n).equals(stateIndex, true))
			difference += sign;

	for(unsigned int n = 0; n < subtractStateIndices.size(); n++)
		if(subtractStateIndices.at(n).equals(stateIndex, true))
			difference -= sign;

	return WrapperRule(
		DifferenceRule(
			addStateIndices,
			subtractStateIndices,
			difference
		)
	);
}

bool DifferenceRule::isSatisfied(
	const FockSpace<BitRegister> &fockSpace,
	const FockState<BitRegister> &fockState
) const{
	int counter = 0;
	for(unsigned int n = 0; n < addStateIndices.size(); n++)
		counter += fockSpace.getSumParticles(fockState, addStateIndices.at(n));

	for(unsigned int n = 0; n < subtractStateIndices.size(); n++)
		counter -= fockSpace.getSumParticles(fockState, subtractStateIndices.at(n));

	return (counter == difference);
}

bool DifferenceRule::isSatisfied(
	const FockSpace<ExtensiveBitRegister> &fockSpace,
	const FockState<ExtensiveBitRegister> &fockState
) const{
	int counter = 0;
	for(unsigned int n = 0; n < addStateIndices.size(); n++)
		counter += fockSpace.getSumParticles(fockState, addStateIndices.at(n));

	for(unsigned int n = 0; n < subtractStateIndices.size(); n++)
		counter -= fockSpace.getSumParticles(fockState, subtractStateIndices.at(n));

	return (counter == difference);
}

bool DifferenceRule::operator==(const FockStateRule &rhs) const{
	switch(rhs.getFockStateRuleID()){
	case FockStateRuleID::WrapperRule:
		//The order is important. Infinite recurison will occur if rhs
		//is on the right.
		return (rhs == *this);
	case FockStateRuleID::DifferenceRule:
		if(addStateIndices.size() != ((const DifferenceRule&)rhs).addStateIndices.size())
			return false;

		if(subtractStateIndices.size() != ((const DifferenceRule&)rhs).subtractStateIndices.size())
			return false;

		if(difference != ((const DifferenceRule&)rhs).difference)
			return false;

		for(unsigned int n = 0; n < addStateIndices.size(); n++)
			if(!addStateIndices.at(n).equals(((const DifferenceRule&)rhs).addStateIndices.at(n), true))
				return false;

		for(unsigned int n = 0; n < subtractStateIndices.size(); n++)
			if(!subtractStateIndices.at(n).equals(((const DifferenceRule&)rhs).subtractStateIndices.at(n), true))
				return false;

		return true;
	default:
		return false;
	}
}

void DifferenceRule::print() const{
	Streams::out << "Sum(";
	for(unsigned int n = 0; n < addStateIndices.size(); n++){
		if(n > 0)
			Streams::out << ", ";

		addStateIndices.at(n).print();
	}
	Streams::out << ") - Sum(";
	for(unsigned int n = 0; n < subtractStateIndices.size(); n++){
		if(n > 0)
			Streams::out << ", ";

		subtractStateIndices.at(n).print();
	}
	Streams::out << ") = " << difference;
}

};	//End of namespace FockStateRule
};	//End of namespace TBTK
