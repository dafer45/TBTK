/* Copyright 2019 Kristofer Björnson
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
 *  @file Real.h
 *  @brief Real number.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_REAL
#define COM_DAFER45_TBTK_REAL

#include "TBTK/PseudoSerializable.h"

namespace TBTK{

/** @brief Real number. */
class Real : PseudoSerializable{
public:
	/** Constructor. */
	Real(){};

	/** Constructor.
	 *
	 *  @param value The value to initilize the Real number with. */
	constexpr Real(double value) : value(value) {}

	/** Constructs an Index from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Real number.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	Real(const std::string &serialization, Serializable::Mode mode);

	/** Type conversion operator. */
	constexpr operator double() const{	return value;	};

	/** Assignment operator.
	 *
	 *  @param value The value to assign the Real.
	 *
	 *  @return The Real after assignment has occured. */
	Real& operator=(double rhs){
		value = rhs;

		return *this;
	}

	/** Addition assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Real after the addition has occured. */
	Real& operator+=(const Real &rhs){
		value += rhs.value;

		return *this;
	}

	/** Subtraction assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Real after the subtraction has occured. */
	Real& operator-=(const Real &rhs){
		value -= rhs.value;

		return *this;
	}

	/** Multiplication assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Real after the multiplication has occured. */
	Real& operator*=(const Real &rhs){
		value *= rhs.value;

		return *this;
	}

	/** Division assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Real after the division has occured. */
	Real& operator/=(const Real &rhs){
		value /= rhs.value;

		return *this;
	}

	/** Serialize Real. Note that Real is PseudoSerializable rather than
	 *  Serializable. This means that the Serializable interface is
	 *  implemented non-virtually.
	 *
	 *  @param mode Serialization mode.
	 *
	 *  @return Serialized string representation of the Real. */
	std::string serialize(Serializable::Mode mode) const;
private:
	/** Value. */
	double value;
};

inline Real::Real(const std::string &serialization, Serializable::Mode mode){
	switch(mode){
	case Serializable::Mode::JSON:
		value = stod(serialization);
		break;
	default:
		TBTKExit(
			"Real::Real()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

inline std::string Real::serialize(Serializable::Mode mode) const{
	switch(mode){
	case Serializable::Mode::JSON:
		return std::to_string(value);
	default:
		TBTKExit(
			"Real::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK

#endif
