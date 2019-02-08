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
 *  @file Natural.h
 *  @brief Natural number.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_NATURAL
#define COM_DAFER45_TBTK_NATURAL

#include "TBTK/PseudoSerializable.h"

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

/** @brief Natural number. */
class Natural : PseudoSerializable{
public:
	/** Constructor. */
	Natural(){};

	/** Constructor.
	 *
	 *  @param value The value to initilize the Natural number with. */
	constexpr Natural(unsigned int value) : value(value) {}

	/** Constructs an Index from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Natural number.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	Natural(const std::string &serialization, Serializable::Mode mode);

	/** Type conversion operator. */
	constexpr operator unsigned int() const{	return value;	};

	/** Assignment operator.
	 *
	 *  @param value The value to assign the Natural.
	 *
	 *  @return The Natural after assignment has occured. */
	Natural& operator=(unsigned int rhs){
		value = rhs;

		return *this;
	}

	/** Addition assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Natural after the addition has occured. */
	Natural& operator+=(const Natural &rhs){
		value += rhs.value;

		return *this;
	}

	/** Subtraction assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Natural after the subtraction has occured. */
	Natural& operator-=(const Natural &rhs){
		value -= rhs.value;

		return *this;
	}

	/** Multiplication assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Natural after the multiplication has occured. */
	Natural& operator*=(const Natural &rhs){
		value *= rhs.value;

		return *this;
	}

	/** Division assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Natural after the division has occured. */
	Natural& operator/=(const Natural &rhs){
		value /= rhs.value;

		return *this;
	}

	/** Increment operator.
	 *
	 *  @return The Natural after the increment has occured. */
	Natural& operator++(){
		value++;

		return *this;
	}

	/** Increment operator.
	 *
	 *  @return The Natural before the increment has occured. */
	Natural operator++(int){
		Natural previous(*this);
		operator++();

		return previous;
	}

	/** Decrement operator.
	 *
	 *  @return The Natural after the decrease has occured. */
	Natural& operator--(){
		value--;

		return *this;
	}

	/** Decrement operator.
	 *
	 *  @return The Natural before the decrease has occured. */
	Natural operator--(int){
		Natural previous(*this);
		operator--();

		return previous;
	}

	/** Serialize Natural. Note that Natural is PseudoSerializable rather
	 *  than Serializable. This means that the Serializable interface is
	 *  implemented non-virtually.
	 *
	 *  @param mode Serialization mode.
	 *
	 *  @return Serialized string representation of the Natural. */
	std::string serialize(Serializable::Mode mode) const;
private:
	/** Value. */
	unsigned int value;
};

inline Natural::Natural(const std::string &serialization, Serializable::Mode mode){
	switch(mode){
	case Serializable::Mode::JSON:
		value = stoul(serialization);
		break;
	default:
		TBTKExit(
			"Natural::Natural()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

inline std::string Natural::serialize(Serializable::Mode mode) const{
	switch(mode){
	case Serializable::Mode::JSON:
		return std::to_string(value);
	default:
		TBTKExit(
			"Natural::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

#else
	typedef unsigned int Natural;
#endif

};	//End of namespace TBTK

#endif
