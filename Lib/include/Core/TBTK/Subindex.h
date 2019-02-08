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

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

/** @brief Subindex number. */
class Subindex : PseudoSerializable{
public:
	/** Constructor.
	 *
	 *  @param value The value to initilize the Subindex with. */
	constexpr Subindex(Integer value) : value(value) {}

	/** Constructs a Subindex from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Subindex.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	Subindex(const std::string &serialization, Serializable::Mode mode);

	/** Assignment operator.
	 *
	 *  @param value The value to assign the Subindex.
	 *
	 *  @return The Subindex after assignment has occured. */
	Subindex& operator=(Integer rhs){
		value = rhs;

		return *this;
	}

	/** Comparison operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return True if the two subindices agree, otherwise false. */
	bool operator==(Subindex rhs) const{
		return value == rhs.value;
	}

	/** Serialize Subindex. Note that Subindex is PseudoSerializable rather
	 *  than Serializable. This means that the Serializable interface is
	 *  implemented non-virtually.
	 *
	 *  @param mode Serialization mode.
	 *
	 *  @return Serialized string representation of the Subindex. */
	std::string serialize(Serializable::Mode mode) const;
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

inline std::string Subindex::serialize(Serializable::Mode mode) const{
	return value.serialize(mode);
}

#else
	typedef int Subindex;
#endif

};	//End of namespace TBTK

#endif
