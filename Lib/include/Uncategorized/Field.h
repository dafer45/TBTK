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
 *  @brief Abstract base class for fields.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FIELD
#define COM_DAFER45_TBTK_FIELD

#include <initializer_list>

namespace TBTK{

/** Field. */
template<typename DataType, typename ArgumentType>
class Field{
public:
	/** Returns the value of the field at the position specified by the argument. */
	virtual const DataType& operator()(std::initializer_list<ArgumentType> argument) const = 0;
private:
};

};	//End namespace TBTK

#endif
