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

#if TBTK_WRAP_PRIMITIVE_TYPES

/** @brief Subindex number. */
class Subindex : PseudoSerializable{
public:
	//TBTKFeature Core.Subindex.Construction.1 2019-09-22
	/** Constructor. */
	Subindex(){};

	//TBTKFeature Core.Subindex.Construction.2.C++ 2019-09-22
	/** Constructor.
	 *
	 *  @param value The value to initilize the Subindex with. */
	constexpr Subindex(Integer value) : value(value) {}

	//TBTKFeature Core.Subindex.Construction.2.C++ 2019-09-22
	/** Constructor.
	 *
	 *  @param value The value to initilize the Subindex with. */
	constexpr Subindex(int value) : value(value) {}

	//TBTKFeature Core.Subindex.Construction.2.C++ 2019-09-22
	/** Constructor.
	 *
	 *  @param value The value to initilize the Subindex with. */
	constexpr Subindex(unsigned int value) : value(value) {}

	//TBTKFeature Core.Subindex.Serialization.1 2019-09-22
	/** Constructs a Subindex from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Subindex.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	Subindex(const std::string &serialization, Serializable::Mode mode);

	//TBTKFeature Core.Subindex.isWildcard.1 2019-09-22
	//TBTKFeature Core.Subindex.isWildcard.2 2019-09-22
	/** Check if the Subindex is a wildcard (IDX_ALL).
	 *
	 *  @return True if the Subindex is a wildcard, otherwise false. */
	bool isWildcard() const;

	//TBTKFeature Core.Subindex.isLabeledWildcard.1 2019-09-22
	//TBTKFeature Core.Subindex.isLabeledWildcard.2 2019-09-22
	/** Check if the Subindex is a labeled wildcard (IDX_ALL_X).
	 *
	 *  @return True if the Subindex is a labeled sildcard, otherwise
	 *  false. */
	bool isLabeledWildcard() const;

	//TBTKFeature Core.Subindex.isSummationIndex.1 2019-09-22
	//TBTKFeature Core.Subindex.isSummationIndex.2 2019-09-22
	/** Check if the Subindex is a summation index (IDX_SUM_ALL).
	 *
	 *  @return True if the Subindex is a summation index, otherwise false.
	 */
	bool isSummationIndex() const;

	/** Check if the Subindex is a range index (IDX_RANGE).
	 *
	 *  @return True if the Subindex is a range index, otherwise false. */
	bool isRangeIndex() const;

	//TBTKFeature Core.Subindex.isSpinIndex.1 2019-09-22
	//TBTKFeature Core.Subindex.isSpinIndex.2 2019-09-22
	/** Check if the Subindex is a spin index (IDX_SPIN).
	 *
	 *  @return True if the SUbindex is a spin index, otherwise false. */
	bool isSpinIndex() const;

	//TBTKFeature Core.Subindex.isIndexSeparator.1 2019-09-22
	//TBTKFeature Core.Subindex.isIndexSeparator.2 2019-09-22
	/** Check if the Subindex is an Index separator (IDX_SEPARATOR).
	 *
	 *  @return True if the Subindex is an Index separator, otherwise
	 *  false. */
	bool isIndexSeparator() const;

	//TBTKFeature Core.Subindex.operatorInt.1.C++ 2019-09-22
	/** Type conversion operator. */
	constexpr operator int() const{ return value;	};

	//TBTKFeature Core.Subindex.operatorAssignment.1.C++ 2019-09-22
	/** Assignment operator.
	 *
	 *  @param value The value to assign the Subindex.
	 *
	 *  @return The Subindex after assignment has occured. */
	Subindex& operator=(Integer rhs){
		value = rhs;

		return *this;
	}

	//TBTKFeature Core.Subindex.operatorAssignment.1.C++ 2019-09-22
	/** Assignment operator.
	 *
	 *  @param value The value to assign the Subindex.
	 *
	 *  @return The Subindex after assignment has occured. */
	Subindex& operator=(int rhs){
		value = rhs;

		return *this;
	}

	//TBTKFeature Core.Subindex.operatorAdditionAssignment.1.C++ 2019-09-22
	/** Addition asignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The subindex after assignment has occured. */
	Subindex& operator+=(Subindex rhs){
		value += rhs.value;

		return *this;
	}

	//TBTKFeature Core.Subindex.operatorSubtractionAssignment.1.C++ 2019-09-22
	/** Subtraction asignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The subindex after assignment has occured. */
	Subindex& operator-=(Subindex rhs){
		value -= rhs.value;

		return *this;
	}

	//TBTKFeature Core.Subindex.operatorMultiplicationAssignment.1.C++ 2019-09-22
	/** Multiplication asignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The subindex after assignment has occured. */
	Subindex& operator*=(Subindex rhs){
		value *= rhs.value;

		return *this;
	}

	//TBTKFeature Core.Subindex.operatorDivisionAssignment.1.C++ 2019-09-22
	/** Division asignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The subindex after assignment has occured. */
	Subindex& operator/=(Subindex rhs){
		value /= rhs.value;

		return *this;
	}

	//TBTKFeature Core.Subindex.operatorPreIncrement.1.C++ 2019-09-22
	/** Increment operator.
	 *
	 *  @return The Subindex after the increment has occured. */
	Subindex& operator++(){
		value++;

		return *this;
	}

	//TBTKFeature Core.Subindex.operatorPostIncrement.1.C++ 2019-09-22
	/** Increment operator.
	 *
	 *  @return The Subindex before the increment has occured. */
	Subindex operator++(int){
		Subindex previous(*this);
		operator++();

		return previous;
	}

	//TBTKFeature Core.Subindex.operatorPreDecrement.1.C++ 2019-09-22
	/** Decrement operator.
	 *
	 *  @return The Subindex after the decrement has occured. */
	Subindex& operator--(){
		value--;

		return *this;
	}

	//TBTKFeature Core.Subindex.operatorPostIncrement.1.C++ 2019-09-22
	/** Decrement operator.
	 *
	 *  @return The Subindex before the decrement has occured. */
	Subindex operator--(int){
		Subindex previous(*this);
		operator--();

		return previous;
	}

	//TBTKFeature Core.Subindex.operatorFunction.1.C++ 2019-09-22
	//TBTKFeature Core.Subindex.operatorFunction.2.C++ 2019-09-22
	/** Function call operator. Used to attach labels to predefined labeled
	 *  Subindices. For example IDX_ALL_(n) can be used to define an nth
	 *  labeled wildcard index. The function call fails if the Index is not
	 *  of a type that supports labels. Note that if the label is so large
	 *  that it has non-zero bits outside of the non-zero bits of
	 *  IDX_FLAG_MASK, the highest valued bits will be truncted.
	 *
	 *  @param label The label.
	 *
	 *  @return A new Subindex with the label attached. */
	Subindex operator()(unsigned int label) const;

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

/** \brief Special subindex values.
 *
 *  Negative Subindex values with the second uppermost bit set to zero are
 *  reserved for flags.<br><br>
 *  <b>IDX_ALL = _a_:</b><br>
 *    Wildcard Used to indicate that all indices are to be considered or that
 *    the particular subindex value is of no interest. To improve
 *    self-documentation for library code, only IDX_ALL should be used in
 *    library code. '_a_' is syntactic sugar meant for use in application code.
 *    <br><br>
 *  <b>IDX_SUM_ALL:</b><br>:
 *    Used to indicate that a subindex should be summed over.<br><br>
 *  <b>IDX_SPIN:</b><br>
 *    Used to indicate that a subindex should be interpreted as a
 *    spin-subindex.<br><br>
 *  <b>IDX_SEPARATOR:</b><br>
 *    Used as Index-separator in compound indices such as {{1, 2}, {3, 4}},
 *    which is stored as {1, 2, IDX_SEPARATOR, 3, 4}.<br><br>
 *  <b>IDX_ALL_ = _aX_</b>:
 *    Labeled wildcards. Are used for the same purpose as wildcard indices
 *    (IDX_ALL), but where two or more subindices should covary. The function
 *    call operator works together with IDX_ALL_ to create wildcards with
 *    different labels. For example,
 *    {IDX_ALL_(0), 1, IDX_ALL_(1), IDX_ALL_(0), IDX_ALL_(1)} indicates that
 *    all indices of the form {m, 1, n, m, n} are of interest. To improve
 *    self-documentation, only IDX_ALL_ should be used in library code. '_aX_'
 *    is syntactic sugar meant for use in application code.<br><br>
 *  <b>IDX_X, IDX_Y, IDX_Z:</b><br>
 *    Loop indices used to indicate that a particular index should be looped
 *    over. */
//_a_ and _aX_ are shorthand notation for IDX_ALL and IDX_ALL_. Never use
//shorthands in library code.
constexpr Subindex IDX_FLAG_MASK	= (int)0x00FFFFFF;
constexpr Subindex IDX_ALL		= (int)(0xBFFFFFFF & ~0x20000000);
constexpr Subindex _a_			= IDX_ALL;
constexpr Subindex IDX_SUM_ALL		= (int)(0xBFFFFFFF & ~0x20000001);
constexpr Subindex IDX_SPIN		= (int)(0xBFFFFFFF & ~0x20000002);
constexpr Subindex IDX_SEPARATOR	= (int)(0xBFFFFFFF & ~0x20000003);
constexpr Subindex IDX_ALL_		= (int)(0xBFFFFFFF & ~0x10000000);
constexpr Subindex _aX_			= IDX_ALL_;
constexpr Subindex IDX_RANGE		= (int)(0xBFFFFFFF & ~0x08000000);
constexpr Subindex IDX_X		= (int)(IDX_RANGE & ~0x00000000);
constexpr Subindex IDX_Y		= (int)(IDX_RANGE & ~0x00000001);
constexpr Subindex IDX_Z		= (int)(IDX_RANGE & ~0x00000002);

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
	return (value | IDX_FLAG_MASK) == IDX_ALL_;
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

inline Subindex Subindex::operator()(unsigned int label) const{
	TBTKAssert(
		value == IDX_ALL_.value,
		"Subindex::operator()",
		"Unsupported subindex type. This function is only supported"
		<< " for the IDX_ALL_ Subindex.",
		""
	);

	return Subindex(value & (-(int)label | ~IDX_FLAG_MASK));
}

inline std::string Subindex::serialize(Serializable::Mode mode) const{
	return value.serialize(mode);
}

#else
	typedef int Subindex;
#endif

};	//End of namespace TBTK

#endif
