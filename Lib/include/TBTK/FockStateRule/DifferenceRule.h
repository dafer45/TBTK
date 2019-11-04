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
 *  @file DifferenceRule.h
 *  @brief DifferenceRule.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DIFFERENCE_RULE
#define COM_DAFER45_TBTK_DIFFERENCE_RULE

#include "TBTK/FockStateRule/FockStateRule.h"
#include "TBTK/Index.h"

namespace TBTK{
namespace FockStateRule{

class DifferenceRule : public FockStateRule{
public:
	/** Constructor */
	DifferenceRule(
		std::initializer_list<Index> addStateIndices,
		std::initializer_list<Index> subtractStateIndices,
		int difference
	);

	/** Constructor */
	DifferenceRule(
		std::vector<Index> addStateIndices,
		std::vector<Index> subtractStateIndices,
		int difference
	);

	/** Destructor. */
	virtual ~DifferenceRule();

	/** Clone DifferenceRule. */
	virtual DifferenceRule* clone() const;

	/** Implements FockStateRule::createNewRule(). */
	virtual WrapperRule createNewRule(
		const LadderOperator<BitRegister> &ladderOperator
	) const;

	/** Implements FockStateRule::createNewRule(). */
	virtual WrapperRule createNewRule(
		const LadderOperator<ExtensiveBitRegister> &ladderOperator
	) const;

	/** Check whether a given FockState fullfills the rule with respect to
	 *  a particular FockSpace. */
	virtual bool isSatisfied(
		const FockSpace<BitRegister> &fockSpace,
		const FockState<BitRegister> &fockState
	) const;

	/** Check whether a given FockState fullfills the rule with respect to
	 *  a particular FockSpace. */
	virtual bool isSatisfied(
		const FockSpace<ExtensiveBitRegister> &fockSpace,
		const FockState<ExtensiveBitRegister> &fockState
	) const;

	/** Comparison operator. */
	virtual bool operator==(const FockStateRule &rhs) const;

	/** Implements FockStateRule::print(). */
	virtual void print() const;
private:
	/** Indices to add. */
	std::vector<Index> addStateIndices;

	/** Indices to subtract. */
	std::vector<Index> subtractStateIndices;

	/** Number of particles that the states corresponding to the indices
	 *  stored in stateIndices are required to sum up to. */
	int difference;
};

};	//End of namespace FockStateRule
};	//End of namespace TBTK

#endif
