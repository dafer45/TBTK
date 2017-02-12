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

#include "SumRule.h"

namespace TBTK{
namespace FockStateRule{

SumRule::SumRule(
	std::initializer_list<Index> stateIndices,
	unsigned int numParticles
){
	for(unsigned int n = 0; n < stateIndices.size(); n++)
		this->stateIndices.push_back(*(stateIndices.begin()+n));

	this->numParticles = numParticles;
}

SumRule::SumRule(
	std::vector<Index> stateIndices,
	unsigned int numParticles
){
	for(unsigned int n = 0; n < stateIndices.size(); n++)
		this->stateIndices.push_back(*(stateIndices.begin()+n));

	this->numParticles = numParticles;
}

SumRule::~SumRule(
){
}

SumRule* SumRule::clone() const{
	return new SumRule(stateIndices, numParticles);
}

bool SumRule::isSatisfied(
	const FockSpace<BitRegister> &fockSpace,
	const FockState<BitRegister> &fockState
) const{
	unsigned int counter = 0;
	for(unsigned int n = 0; n < stateIndices.size(); n++)
		counter += fockSpace.getSumParticles(fockState, stateIndices.at(n));

	return (counter == numParticles);
}

bool SumRule::isSatisfied(
	const FockSpace<ExtensiveBitRegister> &fockSpace,
	const FockState<ExtensiveBitRegister> &fockState
) const{
	unsigned int counter = 0;
	for(unsigned int n = 0; n < stateIndices.size(); n++)
		counter += fockSpace.getSumParticles(fockState, stateIndices.at(n));

	return (counter == numParticles);
}

};	//End of namespace FockStateRule
};	//End of namespace TBTK
