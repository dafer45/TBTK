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
 *  @file Subindex.h
 *  @brief Subindex number.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SUBINDEX
#define COM_DAFER45_TBTK_SUBINDEX

#include "TBTK/Integer.h"
#include "TBTK/Natural.h"
#include "TBTK/PseudoSerializable.h"

#ifndef TBTK_DISABLE_NLOHMANN_JSON
#	include "TBTK/json.hpp"
#endif

namespace TBTK{

/** \brief Enum for special subindex values.
 *
 *  While non-negative subindices in an Index corresponds to normal subindices,
 *  negative subindices are reserved for special purposes.<br><br>
 *  <b>IDX_ALL = _a_:</b><br>
 *    Wildcard Used to indicate that all indices are to be considered or that
 *    the particular subindex value is of no interest. To improve
 *    self-documentation for library code, only IDX_ALL should be used in
 *    actuall library code. '_a_' is syntactic suggar meant for use in
 *    application code.<br><br>
 *  <b>IDX_X, IDX_Y, IDX_Z:</b><br>
 *    Loop indices used to indicate that a particular index should be looped
 *    over.<br><br>
 *  <b>IDX_SPIN:</b><br>
 *    Used to indicate that a certain subindex should be interpreted as a
 *    spin-subindex.<br><br>
 *  <b>IDX_SEPARATOR:</b><br>
 *    Used as Index-separator in compound indices such as {{1, 2}, {3, 4}},
 *    which is stored as {1, 2, IDX_SEPARATOR, 3, 4}. */
enum : int{
	//_a_ and _aX_ are shorthand notation for IDX_ALL and IDX_ALL_X. Never
	//use shorthands in library code.
	IDX_FLAG_MASK	= (int)0x00FFFFFF,
	IDX_ALL		= (int)(0xBFFFFFFF & ~0x20000000),
	_a_		= IDX_ALL,
	IDX_SUM_ALL	= (int)(0xBFFFFFFF & ~0x20000001),
	IDX_SPIN	= (int)(0xBFFFFFFF & ~0x20000002),
	IDX_SEPARATOR	= (int)(0xBFFFFFFF & ~0x20000003),
	IDX_ALL_X	= (int)(0xBFFFFFFF & ~0x10000000),
	IDX_ALL_0	= (int)(IDX_ALL_X & ~0x00000000),
	IDX_ALL_1	= (int)(IDX_ALL_X & ~0x00000001),
	IDX_ALL_2	= (int)(IDX_ALL_X & ~0x00000002),
	_a0_		= IDX_ALL_0,
	_a1_		= IDX_ALL_1,
	_a2_		= IDX_ALL_2,
	IDX_RANGE	= (int)(0xBFFFFFFF & ~0x08000000),
	IDX_X		= (int)(IDX_RANGE & ~0x00000000),
	IDX_Y		= (int)(IDX_RANGE & ~0x00000001),
	IDX_Z		= (int)(IDX_RANGE & ~0x00000002)
};

#if TBTK_WRAP_PRIMITIVE_TYPES

/** @brief Subindex number. */
class Subindex : PseudoSerializable{
public:
	/** Constructor. */
	Subindex(){};

	/** Constructor.
	 *
	 *  @param value The value to initilize the Subindex with. */
	constexpr Subindex(Integer value) : value(value) {}

	/** Constructor.
	 *
	 *  @param value The value to initilize the Subindex with. */
	constexpr Subindex(int value) : value(value) {}

	/** Constructor.
	 *
	 *  @param value The value to initilize the Subindex with. */
	constexpr Subindex(unsigned int value) : value(value) {}

	/** Constructs a Subindex from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Subindex.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	Subindex(const std::string &serialization, Serializable::Mode mode);

	/** Check if the Subindex is a wildcard (IDX_ALL).
	 *
	 *  @return True if the Subindex is a wildcard, otherwise false. */
	bool isWildcard() const;

	/** Check if the Subindex is a labeled wildcard (IDX_ALL_X).
	 *
	 *  @return True if the Subindex is a labeled sildcard, otherwise
	 *  false. */
	bool isLabeledWildcard() const;

	/** Check if the Subindex is a summation index (IDX_SUM_ALL).
	 *
	 *  @return True if the Subindex is a summation index, otherwise false.
	 */
	bool isSummationIndex() const;

	/** Check if the Subindex is a range index (IDX_RANGE).
	 *
	 *  @return True if the Subindex is a range index, otherwise false. */
	bool isRangeIndex() const;

	/** Check if the Subindex is a spin index (IDX_SPIN).
	 *
	 *  @return True if the SUbindex is a spin index, otherwise false. */
	bool isSpinIndex() const;

	/** Check if the Subindex is an Index separator (IDX_SEPARATOR).
	 *
	 *  @return True if the Subindex is an Index separator, otherwise
	 *  false. */
	bool isIndexSeparator() const;

	/** Type conversion operator. */
	constexpr operator int() const{ return value;	};

	/** Assignment operator.
	 *
	 *  @param value The value to assign the Subindex.
	 *
	 *  @return The Subindex after assignment has occured. */
	Subindex& operator=(Integer rhs){
		value = rhs;

		return *this;
	}

	/** Assignment operator.
	 *
	 *  @param value The value to assign the Subindex.
	 *
	 *  @return The Subindex after assignment has occured. */
	Subindex& operator=(int rhs){
		value = rhs;

		return *this;
	}

	/** Addition asignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The subindex after assignment has occured. */
	Subindex& operator+=(Subindex rhs){
		value += rhs.value;

		return *this;
	}

	/** Addition operator.
	 *
	 *  @param lhs The left hand side.
	 *  @param rhs The right hand side.
	 *
	 *  @return The sum of the two subindices. */
/*	friend Subindex operator+(Subindex lhs, Subindex rhs){
		Subindex result(lhs);
		result += rhs;

		return result;
	}*/

	/** Subtraction asignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The subindex after assignment has occured. */
	Subindex& operator-=(Subindex rhs){
		value -= rhs.value;

		return *this;
	}

	/** Subtraction operator.
	 *
	 *  @param lhs The left hand side.
	 *  @param rhs The right hand side.
	 *
	 *  @return The difference between the two subindices. */
/*	friend Subindex operator-(Subindex lhs, Subindex rhs){
		Subindex result(lhs);
		result -= rhs;

		return result;
	}*/

	/** Multiplication asignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The subindex after assignment has occured. */
	Subindex& operator*=(Subindex rhs){
		value *= rhs.value;

		return *this;
	}

	/** Multiplication operator.
	 *
	 *  @param lhs The left hand side.
	 *  @param rhs The right hand side.
	 *
	 *  @return The product of the two subindices. */
/*	friend Subindex operator*(Subindex lhs, Subindex rhs){
		Subindex result(lhs);
		result *= rhs;

		return result;
	}*/

	/** Division asignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The subindex after assignment has occured. */
	Subindex& operator/=(Subindex rhs){
		value /= rhs.value;

		return *this;
	}

	/** Division operator.
	 *
	 *  @param lhs The left hand side.
	 *  @param rhs The right hand side.
	 *
	 *  @return The quotient between the two subindices. */
/*	friend Subindex operator/(Subindex lhs, Subindex rhs){
		Subindex result(lhs);
		result /= rhs;

		return result;
	}*/

	/** Increment operator.
	 *
	 *  @return The Subindex after the increment has occured. */
	Subindex& operator++(){
		value++;

		return *this;
	}

	/** Increment operator.
	 *
	 *  @return The Subindex before the increment has occured. */
	Subindex operator++(int){
		Subindex previous(*this);
		operator++();

		return previous;
	}

	/** Decrement operator.
	 *
	 *  @return The Subindex after the decrement has occured. */
	Subindex& operator--(){
		value--;

		return *this;
	}

	/** Decrement operator.
	 *
	 *  @return The Subindex before the decrement has occured. */
	Subindex operator--(int){
		Subindex previous(*this);
		operator--();

		return previous;
	}

	/** Comparison operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return True if the two subindices agree, otherwise false. */
/*	bool operator==(Subindex rhs) const{
		return value == rhs.value;
	}*/

	/** Not equal operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return False if the two subindices agree, otherwise true. */
/*	bool operator!=(Subindex rhs) const{
		return !operator==(rhs);
	}*/

	/** Less than operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return True if the left hand side is smaller than the right hand
	 *  side, otherwise false. */
/*	bool operator<(Subindex rhs) const{
		return value < rhs.value;
	}*/

	/** Less than operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return True if the left hand side is smaller than the right hand
	 *  side, otherwise false. */
/*	friend bool operator<(int lhs, Subindex rhs){
		return lhs < rhs.value;
	}*/

	/** Larger than operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return True if the left hand side is larger than the right hand
	 *  side, otherwise false */
/*	bool operator>(Subindex rhs) const{
		return value > rhs.value;
	}*/

	/** Larger than operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return True if the left hand side is larger than the right hand
	 *  side, otherwise false */
/*	friend bool operator>(int lhs, Subindex rhs){
		return lhs > rhs.value;
	}*/

	/** Less or equal to operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return True if the left hand side is less or equal to the right
	 *  hand side, otherwise false. */
/*	bool operator<=(Subindex rhs) const{
		return value <= rhs.value;
	}*/

	/** Less or equal to operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return True if the left hand side is less or equal to the right
	 *  hand side, otherwise false. */
/*	friend bool operator<=(int lhs, Subindex rhs){
		return lhs <= rhs.value;
	}*/

	/** Larger or equal to operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return True if the left hand side is larger or equal to the right
	 *  hand side, otherwise false. */
/*	bool operator>=(Subindex rhs) const{
		return value >= rhs.value;
	}*/

	/** Larger or equal to operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return True if the left hand side is larger or equal to the right
	 *  hand side, otherwise false. */
/*	friend bool operator>=(int lhs, Subindex rhs){
		return lhs >= rhs.value;
	}*/

	/** ostream operator.
	 *
	 *  @param os The ostream to write to.
	 *  @param rhs The Subindex to write.
	 *
	 *  @return The ostream. */
	friend std::ostream& operator<<(std::ostream &os, const Subindex subindex){
		os << subindex.value;

		return os;
	}

	/** istream operator.
	 *
	 *  @param os The istream to read from.
	 *  @param rhs The Subindex to read to.
	 *
	 *  @return The istream. */
	friend std::istream& operator>>(std::istream &is, Subindex &subindex){
		int i;
		is >> i;
		subindex.value = Integer(i);

		return is;
	}

	/** Serialize Subindex. Note that Subindex is PseudoSerializable rather
	 *  than Serializable. This means that the Serializable interface is
	 *  implemented non-virtually.
	 *
	 *  @param mode Serialization mode.
	 *
	 *  @return Serialized string representation of the Subindex. */
	std::string serialize(Serializable::Mode mode) const;

#ifndef TBTK_DISABLE_NLOHMANN_JSON
	/** Implements the Nlohmann json interface for conversion to json.
	 *
	 *  @param j The json output.
	 *  @param subindex The Subindex to convert. */
	friend void to_json(nlohmann::json &j, const Subindex &subindex){
		to_json(j, subindex.value);
	}

	/** Implements the Nlohmann json interface for conversion from json.
	 *
	 *  @param j The json input.
	 *  @param subindex The Subindex to convert to. */
	friend void from_json(const nlohmann::json &j, Subindex &subindex){
		from_json(j, subindex.value);
	}
#endif
private:
	/** Value. */
	Integer value;
};

inline Subindex::Subindex(
	const std::string &serialization,
	Serializable::Mode mode
) :
	value(serialization, mode)
{
}

inline bool Subindex::isWildcard() const{
	return value == IDX_ALL;
}

inline bool Subindex::isLabeledWildcard() const{
	return (value | IDX_FLAG_MASK) == IDX_ALL_X;
}

inline bool Subindex::isSummationIndex() const{
	return value == IDX_SUM_ALL;
}

inline bool Subindex::isRangeIndex() const{
	return (value | IDX_FLAG_MASK) == IDX_RANGE;
}

inline bool Subindex::isSpinIndex() const{
	return value == IDX_SPIN;
}

inline bool Subindex::isIndexSeparator() const{
	return value == IDX_SEPARATOR;
}

inline std::string Subindex::serialize(Serializable::Mode mode) const{
	return value.serialize(mode);
}

#else
	typedef int Subindex;
#endif

};	//End of namespace TBTK

#endif
