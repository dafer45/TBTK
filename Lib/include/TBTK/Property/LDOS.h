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
 *  @file LDOS.h
 *  @brief Property container for local density of states (LDOS).
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_LDOS
#define COM_DAFER45_TBTK_LDOS

#include "TBTK/Property/EnergyResolvedProperty.h"

namespace TBTK{
namespace Property{

/** @brief Property container for the local density of states (LDOS).
 *
 *  The LDOS is an EnergyResolvedProperty with DataType double.
 *
 *  # Convention
 *  LDOS extracted by native PropertyExtractors satisfies the following
 *  convention.
 *
 *  - The LDOS is normalized such that, if it covers the full energy range, it
 *  integrates to one for any given Index. \f$\sum_n LDOS(n)\Delta E = 1\f$.
 *
 *  # Example
 *  \snippet Property/LDOS.cpp LDOS
 *  ## Output
 *  \snippet output/Property/LDOS.txt LDOS
 *  \image html output/Property/LDOS/figures/PropertyLDOSLDOS0.png
 *  \image html output/Property/LDOS/figures/PropertyLDOSLDOS1.png */
class LDOS : public EnergyResolvedProperty<double>{
public:
	/** Constructs an uninitialized LDOS. */
	LDOS();

	/** Constructs LDOS on the Ranges format. [See AbstractProperty for
	 *  detailed information about the Ranges format.]
	 *
	 *  @param ranges The upper limit (exclusive) for the corresponding
	 *  dimensions.
	 *
	 *  @param energyWindow The energy window over which the LDOS is
	 *  defined.
	 *
	 *  @param upperBound Upper bound for the energy
	 *  @param resolution Number of points to use for the energy. */
	LDOS(const std::vector<int> &ranges, const Range &energyWindow);

	/** Constructs LDOS on the Ranges format and initializes it with data.
	 *  [See AbstractProperty for detailed information about the Ranges
	 *  format and the raw data format.]
	 *
	 *  @param ranges The upper limit (exclusive) for the corresponding
	 *  dimensions.
	 *
	 *  @param energyWindow The energy window over which the LDOS is
	 *  defined.
	 *
	 *  @param data Raw data to initialize the LDOS with. */
	LDOS(
		const std::vector<int> &ranges,
		const Range &energyWindow,
		const double *data
	);

	/** Constructs LDOS on the Custom format. [See AbstractProperty for
	 *  detailed information about the Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the LDOS should be contained.
	 *
	 *  @param energyWindow The energy window over which the LDOS is
	 *  defined. */
	LDOS(const IndexTree &indexTree, const Range &energyWindow);

	/** Constructs LDOS on the Custom format and initializes it with data.
	 *  [See AbstractProperty for detailed information about the Custom
	 *  format and the raw data format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the LDOS should be contained.
	 *
	 *  @param energyWindow The energy window over which the LDOS is
	 *  defined.
	 *
	 *  @param data Raw data to initialize the LDOS with. */
	LDOS(
		const IndexTree &indexTree,
		const Range &energyWindow,
		const double *data
	);

	/** Constructor. Construct the LDOS from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the LDOS.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	LDOS(const std::string &serialization, Mode mode);

	/** Overrides EnergyResolvedProperty::operator+=(). */
	LDOS& operator+=(const LDOS &rhs);

	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new LDOS that is the sum of this LDOS and the right hand
	 *  side. */
	LDOS operator+(const LDOS &rhs) const;

	/** Overrides EnergyResolvedProperty::operator-=(). */
	LDOS& operator-=(const LDOS &rhs);

	/** Subtraction operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new LDOS that is the difference between this LDOS and the
	 *  right hand side. */
	LDOS operator-(const LDOS &rhs) const;

	/** Overrides EnergyResolvedProperty::operator*=(). */
	LDOS& operator*=(const double &rhs);

	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new LDOS that is product of this LDOS and the right hand
	 *  side. */
	LDOS operator*(const double &rhs) const;

	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new LDOS that is product of the left hand side and the
	 *  LDOS. */
	friend LDOS operator*(const double &lhs, const LDOS &rhs){
		return rhs*lhs;
	}

	/** Overrides EnergyResolvedProperty::operator/=(). */
	LDOS& operator/=(const double &rhs);

	/** Division operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new LDOS that is quotient between this LDOS and the right
	 *  hand side. */
	LDOS operator/(const double &rhs) const;

	/** Implements Streamable::toString(). */
	virtual std::string toString() const;

	/** Overrides AbstractProperty::serialize(). */
	virtual std::string serialize(Mode mode) const;
private:
};

inline LDOS::LDOS(){
}

inline LDOS& LDOS::operator+=(const LDOS &rhs){
	EnergyResolvedProperty::operator+=(rhs);

	return *this;
}

inline LDOS LDOS::operator+(const LDOS &rhs) const{
	LDOS ldos = *this;

	return ldos += rhs;
}

inline LDOS& LDOS::operator-=(const LDOS &rhs){
	EnergyResolvedProperty::operator-=(rhs);

	return *this;
}

inline LDOS LDOS::operator-(const LDOS &rhs) const{
	LDOS ldos = *this;

	return ldos -= rhs;
}

inline LDOS& LDOS::operator*=(const double &rhs){
	EnergyResolvedProperty::operator*=(rhs);

	return *this;
}

inline LDOS LDOS::operator*(const double &rhs) const{
	LDOS ldos = *this;

	return ldos *= rhs;
}

inline LDOS& LDOS::operator/=(const double &rhs){
	EnergyResolvedProperty::operator/=(rhs);

	return *this;
}

inline LDOS LDOS::operator/(const double &rhs) const{
	LDOS ldos = *this;

	return ldos /= rhs;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
