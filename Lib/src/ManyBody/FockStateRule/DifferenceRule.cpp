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

#include "DifferenceRule.h"
#include "FockSpace.h"

namespace TBTK{
namespace FockStateRule{

DifferenceRule::DifferenceRule(
	std::initializer_list<Index> addStateIndices,
	std::initializer_list<Index> subtractStateIndices,
	int difference
){
	for(unsigned int n = 0; n < addStateIndices.size(); n++)
		this->addStateIndices.push_back(*(addStateIndices.begin()+n));

	for(unsigned int n = 0; n < subtractStateIndices.size(); n++)
		this->subtractStateIndices.push_back(*(subtractStateIndices.begin()+n));

	this->difference = difference;
}

DifferenceRule::DifferenceRule(
	std::vector<Index> addStateIndices,
	std::vector<Index> subtractStateIndices,
	int difference
){
	for(unsigned int n = 0; n < addStateIndices.size(); n++)
		this->addStateIndices.push_back(*(addStateIndices.begin()+n));

	for(unsigned int n = 0; n < subtractStateIndices.size(); n++)
		this->subtractStateIndices.push_back(*(subtractStateIndices.begin()+n));

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

};	//End of namespace FockStateRule
};	//End of namespace TBTK
