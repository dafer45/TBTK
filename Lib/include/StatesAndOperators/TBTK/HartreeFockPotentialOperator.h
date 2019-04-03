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

/** @package TBTKcalc
 *  @file HartreeFockPotentialOperator.h
 *  @brief Hartree-Fock potential operator.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_HARTREE_FOCK_POTENTIAL_OPERATOR
#define COM_DAFER45_TBTK_HARTREE_FOCK_POTENTIAL_OPERATOR

#include "TBTK/AbstractOperator.h"

namespace TBTK{

class HartreeFockPotentialOperator : public AbstractOperator{
public:
	/** Constructor.
	 *
	 *  @param firstState The first state associated with the operator.
	 *  @param secondState The second state associated with the operator.
	 */
	HartreeFockPotentialOperator(
		const AbstractState &firstState,
		const AbstractState &secondState
	);

	/** Copy constructor.
	 *
	 *  @param hartreeFockPotentialOperator HartreeFockPotentialOperator to
	 *  copy. */
	HartreeFockPotentialOperator(
		const HartreeFockPotentialOperator &hartreeFockPotentialOperator
	);

	/** Destructor. */
	~HartreeFockPotentialOperator();

	/** Assignment operator.
	 *
	 *  @param rhs HartreeFockPotentialOperator to assign to the left hand
	 *  side.
	 *
	 *  @return Reference to the assigned HartreeFockPotentialOperator. */
	HartreeFockPotentialOperator& operator=(
		const HartreeFockPotentialOperator &rhs
	);

	/** Get the first state.
	 *
	 *  @return The first state associated with the operator. */
	const AbstractState& getFirstState() const;

	/** Get the second state.
	 *
	 *  @return The second state associated with the operator. */
	const AbstractState& getSecondState() const;
private:
	/** The first state associated with the operator. */
	AbstractState *firstState;

	/** The second state associated with the operator. */
	AbstractState *secondState;
};

inline HartreeFockPotentialOperator::HartreeFockPotentialOperator(
	const AbstractState &firstState,
	const AbstractState &secondState
) :
	AbstractOperator(OperatorID::HartreeFockPotential),
	firstState(firstState.clone()),
	secondState(secondState.clone())
{
}

inline HartreeFockPotentialOperator::HartreeFockPotentialOperator(
	const HartreeFockPotentialOperator &hartreeFockPotentialOperator
) :
	AbstractOperator(hartreeFockPotentialOperator)
{
	firstState = hartreeFockPotentialOperator.firstState->clone();
	secondState = hartreeFockPotentialOperator.secondState->clone();
}

inline HartreeFockPotentialOperator::~HartreeFockPotentialOperator(){
	delete firstState;
	delete secondState;
}

inline HartreeFockPotentialOperator& HartreeFockPotentialOperator::operator=(
	const HartreeFockPotentialOperator &rhs
){
	if(this != &rhs){
		firstState = rhs.firstState->clone();
		secondState = rhs.secondState->clone();
	}

	return *this;
}

inline const AbstractState&
HartreeFockPotentialOperator::getFirstState() const{
	return *firstState;
}

inline const AbstractState&
HartreeFockPotentialOperator::getSecondState() const{
	return *secondState;
}

};	//End of namespace TBTK

#endif
