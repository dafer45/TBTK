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
 *  @file Density.h
 *  @brief Property container for density.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DENSITY
#define COM_DAFER45_TBTK_DENSITY

#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/IndexTree.h"

namespace TBTK{
namespace Property{

/** @brief Property container for density.
 *
 *  The Density is a @link AbstractProperty Property@endlink with DataType
 *  double and is defined for a number of @link Index Indices@endlink.
 *
 *  # Example
 *  \snippet Property/Density.cpp Density
 *  ## Output
 *  \snippet output/Property/Density.output Density
 *  \image html output/Property/Density/figures/PropertyDensityDensity.png
 *  \image html output/Property/Density/figures/PropertyDensityDensityCut.png */
class Density : public AbstractProperty<double>{
public:
	/** Constructs an uninitialized Density. */
	Density();

	/** Constructs a Density on the Ranges format. [See AbstractProperty
	 *  for detailed information about the Ranges format.]
	 *
	 *  @param ranges The upper limits (exclusive) for the corresponding
	 *  dimensions. */
	Density(const std::vector<int> &ranges);

	/** Constructs a Density on the Ranges format and initializes it with
	 *  data. [See AbstractProperty for detailed information about the
	 *  Ranges format and the raw data format.]
	 *
	 *  @param ranges The upper limits (exclusive) for the corresponding
	 *  dimensions.
	 *
	 *  @param data Raw data to initialize the Density with. */
	Density(const std::vector<int> &ranges, const double *data);

	/** Constructs a Density on the Custom format. [See AbstractProperty
	 *  for detailed information about the Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the Density should be contained. */
	Density(const IndexTree &indexTree);

	/** Constructs a Density on the Custom format. [See AbstractProperty
	 *  for detailed information about the Custom format and the raw data
	 *  format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the Density should be contained.
	 *
	 *  @param data Raw data to initialize the Density with. */
	Density(const IndexTree &indexTree, const double *data);

	/** Constructor. Constructs the Density from a serializeation string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Density.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	Density(const std::string &serialization, Mode mode);

	/** Overrides AbstractPropertye::operator+=(). */
	Density& operator+=(const Density &rhs);

	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new Density that is the sum of this Density and the right
	 *  hand side. */
	Density operator+(const Density &rhs) const;

	/** Overrides AbstractPropertye::operator-=(). */
	Density& operator-=(const Density &rhs);

	/** Subtraction operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new Density that is the difference between this Density
	 *  and the right hand side. */
	Density operator-(const Density &rhs) const;

	/** Overrides AbstractPropertye::operator*=(). */
	Density& operator*=(const double &rhs);

	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new Density that is the product of the Density and the
	 *  right hand side. */
	Density operator*(const double &rhs) const;

	/** Multiplication operator.
	 *
	 *  @param lhs The left hand side of the equation.
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new Density that is the product of the left hand side and
	 *  the Density. */
	friend Density operator*(const double &lhs, const Density &rhs){
		return rhs*lhs;
	}

	/** Overrides AbstractPropertye::operator/=(). */
	Density& operator/=(const double &rhs);

	/** Division operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new Density that is the quotient of the Density and the
	 *  right hand side. */
	Density operator/(const double &rhs) const;

	/** Get the minimum value for the Density.
	 *
	 *  @return The minimum value.*/
	double getMin() const;

	/** Get maximum value for the Density.
	 *
	 *  @return The maximum value. */
	double getMax() const;

	/** Overrides Streamable::toString(). */
	virtual std::string toString() const;

	/** Overrides AbstractProperty::serialize(). */
	virtual std::string serialize(Mode mode) const;
private:
};

inline Density::Density(){
}

inline Density& Density::operator+=(const Density &rhs){
	AbstractProperty::operator+=(rhs);

	return *this;
}

inline Density Density::operator+(const Density &rhs) const{
	Density density = *this;

	return density += rhs;
}

inline Density& Density::operator-=(const Density &rhs){
	AbstractProperty::operator-=(rhs);

	return *this;
}

inline Density Density::operator-(const Density &rhs) const{
	Density density = *this;

	return density -= rhs;
}

inline Density& Density::operator*=(const double &rhs){
	AbstractProperty::operator*=(rhs);

	return *this;
}

inline Density Density::operator*(const double &rhs) const{
	Density density = *this;

	return density *= rhs;
}

inline Density& Density::operator/=(const double &rhs){
	AbstractProperty::operator/=(rhs);

	return *this;
}

inline Density Density::operator/(const double &rhs) const{
	Density density = *this;

	return density /= rhs;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
