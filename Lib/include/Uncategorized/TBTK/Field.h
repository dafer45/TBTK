/* Copyright 2017 Kristofer Björnson
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
 *  @file Field.h
 *  @brief Abstract base class for fields.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FIELD
#define COM_DAFER45_TBTK_FIELD

#include "TBTK/TBTKMacros.h"

#include <initializer_list>

namespace TBTK{

/** Field. */
template<typename DataType, typename ArgumentType>
class Field{
public:
	/** Constructor. */
	Field(bool isCompact = false);

	/** Returns the value of the field at the position specified by the argument. */
	virtual DataType operator()(std::initializer_list<ArgumentType> argument) const = 0;

	/** Returns true if the field is compact. A field that is impact should
	 *  override getCoordinates() and getExtent(). */
	bool getIsCompact() const;

	/** Get coordinates. */
//	virtual const std::vector<double>& getCoordinates() const = 0;
	virtual const std::vector<ArgumentType>& getCoordinates() const;

	/** Get the radial extent of the field. */
//	virtual double getExtent() const = 0;
	virtual ArgumentType getExtent() const;
private:
	/** Flag indicating whether the field is compact. */
	bool isCompact;
};

template<typename DataType, typename ArgumentType>
Field<DataType, ArgumentType>::Field(bool isCompact){
	this->isCompact = isCompact;
}

template<typename DataType, typename ArgumentType>
bool Field<DataType, ArgumentType>::getIsCompact() const{
	return isCompact;
}

template<typename DataType, typename ArgumentType>
const std::vector<ArgumentType>& Field<DataType, ArgumentType>::getCoordinates() const{
	TBTKExit(
		"Field::getCoordinates()",
		"The Field is not compact.",
		""
	);
}

template<typename DataType, typename ArgumentType>
ArgumentType Field<DataType, ArgumentType>::getExtent() const{
	TBTKExit(
		"Field::getCoordinates()",
		"Field is not compact.",
		""
	);
}

};	//End namespace TBTK

#endif
/// @endcond
