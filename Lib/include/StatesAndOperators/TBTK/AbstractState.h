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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file AbstractState.h
 *  @brief Abstract state class from which other states inherit.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ABSTRACT_STATE
#define COM_DAFER45_TBTK_ABSTRACT_STATE

#include "TBTK/AbstractOperator.h"
#include "TBTK/DefaultOperator.h"
#include "TBTK/Index.h"
#include "TBTK/Serializable.h"

#include <complex>
#include <initializer_list>
#include <limits>
#include <vector>

namespace TBTK{

class AbstractState : public Serializable{
public:
	/** List of state identifiers. Officially supported operators are given
	 *  unique identifiers. Operators not (yet) supported should make sure
	 *  they use an identifier that does not clash with the officially
	 *  supported ones [ideally a large random looking number (magic
	 *  number) to also minimize accidental clashes with other operators
	 *  that are not (yet) supported]. */
	enum StateID{
		Basic = 0,
		STO3G = 1,
		Gaussian = 2
	};

	/** Constructor. */
	AbstractState(StateID stateID);

	/** Destructor. */
	virtual ~AbstractState();

	/** Returns a pointer to a clone of the State. */
	virtual AbstractState* clone() const = 0;

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
		const AbstractOperator &o = DefaultOperator()
	) const = 0;

	/** Get state identifier. */
	StateID getStateID() const;

	/** Set coordinates. */
	void setCoordinates(std::initializer_list<double> coordinates);

	/** Set coordinates. */
	void setCoordinates(const std::vector<double> &coordinates);

	/** Set specifiers. */
	void setSpecifiers(std::initializer_list<int> specifiers);

	/** Set specifiers. */
	void setSpecifiers(const std::vector<int> &specifiers);

	/** Set index. */
	void setIndex(const Index &index);

	/** Set container. (For example a unit cell index.) */
	void setContainer(const Index &container);

	/** Set radial extent. */
	void setExtent(double extent);

	/** Get coordinates. */
	const std::vector<double>& getCoordinates() const;

	/** Get specifiers. */
	const std::vector<int>& getSpecifiers() const;

	/** Get index. */
	const Index& getIndex() const;

	/** Get container. */
	const Index& getContainer() const;

	/** Get radial extent. */
	double getExtent() const;

	/** Returns true if the state has finite extent. */
	bool hasFiniteExtent() const;

	/** Implements Serializable::serialize(). */
	std::string serialize(Mode mode) const;
private:
	/** State identifier. */
	StateID stateID;

	/** Coordinates. */
	std::vector<double> coordinates;

	/** Specifiers such as orbital number, spin-species, etc. */
	std::vector<int> specifiers;

	/** Index of the state. */
	Index index;

	/** Index specifiyng the container of the state. For example, a unit
	 *  cell index. */
	Index container;

	/** Spatial radial extent of the state. */
	double extent;
};

inline AbstractState::StateID AbstractState::getStateID() const{
	return stateID;
}

inline void AbstractState::setIndex(const Index& index){
	this->index = index;
}

inline void AbstractState::setContainer(const Index& container){
	this->container = container;
}

inline void AbstractState::setExtent(double extent){
	this->extent = extent;
}

inline const std::vector<double>& AbstractState::getCoordinates() const{
	return coordinates;
}

inline const std::vector<int>& AbstractState::getSpecifiers() const{
	return specifiers;
}

inline const Index& AbstractState::getIndex() const{
	return index;
}

inline const Index& AbstractState::getContainer() const{
	return container;
}

inline double AbstractState::getExtent() const{
	return extent;
}

inline bool AbstractState::hasFiniteExtent() const{
	if(std::numeric_limits<double>::has_infinity)
		return !(extent == std::numeric_limits<double>::infinity());
	else
		return !(extent == std::numeric_limits<double>::max());
}

};	//End of namespace TBTK

#endif
/// @endcond
