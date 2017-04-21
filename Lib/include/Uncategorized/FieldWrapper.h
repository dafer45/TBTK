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

/** @package TBTKcalc
 *  @file Field.h
 *  @brief Wrapper class for a Field.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FIELD_WRAPPER
#define COM_DAFER45_TBTK_FIELD_WRAPPER

#include "Field.h"

#include <initializer_list>

namespace TBTK{

/** FieldWrapper. */
class FieldWrapper{
public:
	/** Enum class for describing the data type. */
	enum class DataType{
		ComplexDouble
	};

	/** Enum class for describing the argument type. */
	enum class ArgumentType{
		Double
	};

	/** Constructor. */
	template<typename DataType, typename ArgumentType>
	FieldWrapper(Field<DataType, ArgumentType> &field);
private:
	/** Pointer to field. */
	void *field;

	/** Data type. */
	DataType dataType;

	/** Argument type. */
	ArgumentType argumentType;
};

template<>
inline FieldWrapper::FieldWrapper(Field<std::complex<double>, double> &field){
	this->field = &field;
	dataType = DataType::ComplexDouble;
	argumentType = ArgumentType::Double;
}

};	//End namespace TBTK

#endif
