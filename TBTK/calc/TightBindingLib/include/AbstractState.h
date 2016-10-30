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
 *  @file AbstractState.h
 *  @brief Abstract state class from which other states inherit.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ABSTRACT_STATE
#define COM_DAFER45_TBTK_ABSTRACT_STATE

#include "AbstractOperator.h"
#include "DefaultOperator.h"

#include <complex>
#include <vector>
#include <initializer_list>

namespace TBTK{

class AbstractState{
public:
	/** List of state identifiers. Officially supported operators are given
	 *  unique identifiers. Operators not (yet) supported should make sure
	 *  they use an identifier that does not clash with the officially
	 *  supported ones [ideally a large random looking number (magic
	 *  number) to also minimize accidental clashes with other operators
	 *  that are not (yet) supported]. */
	enum StateID{
		Basic = 0
	};

	/** Constructor. */
	AbstractState(StateID stateID);

	/** Destructor. */
	virtual ~AbstractState();

	/** Pure virtual overlap function. Returns the value of the operation
	 *  \f[\langle\Psi_1|\Psi_2\rangle\f], where \f[\Psi_1\f] and
	 *  \f[\Psi_2\f] are the argument bra and the object itself,
	 *  respectively. */
	virtual std::complex<double> getOverlap(const AbstractState &ket) const = 0;

	/** Pure virtual matrix element function. Returns the value of the11
	 *  operation \f[\langle\Psi_1|o|\Psi_2\rangle\f], where \f[\Psi_1\f]
	 *  and \f[\Psi_2\f] are the argument bra and the object itself,
	 *  respectively, and o is an operator. */
	virtual std::complex<double> getMatrixElement(
		const AbstractState &ket,
		const AbstractOperator &o
	) const = 0;

	/** Get state identifier. */
	StateID getStateID() const;

	/** Set coordinates. */
	void setCoordinates(std::initializer_list<double> coordinates);

	/** Set specifiers. */
	void setSpecifiers(std::initializer_list<int> specifiers);

	/** Get coordinates. */
	const std::vector<double>& getCoordinates() const;

	/** Get specifiers. */
	const std::vector<int>& getSpecifiers() const;
private:
	/** State identifier. */
	StateID stateID;

	/** Coordinates. */
	std::vector<double> coordinates;

	/** Specifiers such as orbital number, spin-species, etc. */
	std::vector<int> specifiers;
};

inline AbstractState::StateID AbstractState::getStateID() const{
	return stateID;
}

inline const std::vector<double>& AbstractState::getCoordinates() const{
	return coordinates;
}

inline const std::vector<int>& AbstractState::getSpecifiers() const{
	return specifiers;
}

};	//End of namespace TBTK

#endif
