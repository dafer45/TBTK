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

/** @package TBTKcalc
 *  @file FockStateSumRule.h
 *  @brief FockStateSumRule.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SUM_RULE
#define COM_DAFER45_TBTK_SUM_RULE

#include "FockStateRule.h"
#include "Index.h"

#include <initializer_list>
#include <vector>

namespace TBTK{
namespace FockStateRule{

template<typename BIT_REGISTER>
class SumRule : public FockStateRule<BIT_REGISTER>{
public:
	/** Constructor */
	SumRule(
		std::initializer_list<Index> stateIndices,
		unsigned int numParticles
	);

	/** Constructor */
	SumRule(
		std::vector<Index> stateIndices,
		unsigned int numParticles
	);

	/** Destructor. */
	~SumRule();

	/** Check whether a given FockState fullfills the rule with respect to
	 *  a particular FockSpace. */
	virtual bool isFulfilled(
		const FockSpace<BIT_REGISTER> &fockSpace,
		const FockState<BIT_REGISTER> &fockState
	);
private:
	/** Indices to sum over. */
	std::vector<Index> stateIndices;

	/** Number of particles that the states corresponding to the indices
	 *  stored in stateIndices are required to sum up to. */
	unsigned int numParticles;
};

template<typename BIT_REGISTER>
SumRule<BIT_REGISTER>::SumRule(
	std::initializer_list<Index> stateIndices,
	unsigned int numParticles
){
	for(unsigned int n = 0; n < stateIndices.size(); n++)
		this->stateIndices.push_back(*(stateIndices.begin()+n));

	this->numParticles = numParticles;
}

template<typename BIT_REGISTER>
SumRule<BIT_REGISTER>::SumRule(
	std::vector<Index> stateIndices,
	unsigned int numParticles
){
	for(unsigned int n = 0; n < stateIndices.size(); n++)
		this->stateIndices.push_back(*(stateIndices.begin()+n));

	this->numParticles = numParticles;
}

template<typename BIT_REGISTER>
SumRule<BIT_REGISTER>::~SumRule(
){
}

template<typename BIT_REGISTER>
bool SumRule<BIT_REGISTER>::isFulfilled(
	const FockSpace<BIT_REGISTER> &fockSpace,
	const FockState<BIT_REGISTER> &fockState
){
	unsigned int counter = 0;
	for(unsigned int n = 0; n < stateIndices.size(); n++)
		counter += fockSpace.getSumParticles(fockState, stateIndices.at(n));

	return (counter == numParticles);
}

};	//End of namespace FockStateRule
};	//End of namespace TBTK

#endif
