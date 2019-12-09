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
 *  @brief Wrapper class for a Field.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FIELD_WRAPPER
#define COM_DAFER45_TBTK_FIELD_WRAPPER

#include "TBTK/Field.h"

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

	/** Get data type. */
	DataType getDataType() const;

	/** Get argument type. */
	ArgumentType getArgumentType() const;

	/** Function call operator wrapping Field::operator(). */
	template<typename Data, typename Argument>
	Data operator()(std::initializer_list<Argument> arguments) const;

	/** Returns true if the wrapped field is compact. */
	template<typename Data, typename Argument>
	bool getIsCompact() const;

	/** Get coordinates. */
	template<typename Data, typename Argument>
	const std::vector<Argument>& getCoordinates() const;
//	const std::vector<double>& getCoordinates() const;

	/** Wrapping Field::getExtent(). */
	template<typename Data, typename Argument>
	Argument getExtent() const;
//	double getExtent() const;
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

inline FieldWrapper::DataType FieldWrapper::getDataType() const{
	return dataType;
}

inline FieldWrapper::ArgumentType FieldWrapper::getArgumentType() const{
	return argumentType;
}

template<>
inline std::complex<double> FieldWrapper::operator()<std::complex<double>, double>(
	std::initializer_list<double> arguments
) const{
	return ((Field<std::complex<double>, double>*)field)->operator()(arguments);
}

template<>
inline bool FieldWrapper::getIsCompact<std::complex<double>, double>() const{
	return ((Field<std::complex<double>, double>*)field)->getIsCompact();
}

template<>
inline const std::vector<double>& FieldWrapper::getCoordinates<std::complex<double>, double>() const{
	return ((Field<std::complex<double>, double>*)field)->getCoordinates();
}

template<>
inline double FieldWrapper::getExtent<std::complex<double>, double>() const{
	return ((Field<std::complex<double>, double>*)field)->getExtent();
}

};	//End namespace TBTK

#endif
/// @endcond
