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

/** @file EigenValues.cpp
 *
 *  @author Kristofer Björnson
*/

#include "EigenValues.h"

namespace TBTK{
namespace Property{

EigenValues::EigenValues(
	int size
) :
	AbstractProperty(size)
{
}

EigenValues::EigenValues(
	int size,
	const double *data
) :
	AbstractProperty(size, data)
{
}

EigenValues::EigenValues(
	const EigenValues &eigenValues
) :
	AbstractProperty(eigenValues)
{
}

EigenValues::EigenValues(
	EigenValues &&eigenValues
) :
	AbstractProperty(std::move(eigenValues))
{
}

EigenValues::~EigenValues(){
}

EigenValues& EigenValues::operator=(const EigenValues &rhs){
	if(this != &rhs)
		AbstractProperty::operator=(rhs);

	return *this;
}

EigenValues& EigenValues::operator=(EigenValues &&rhs){
	if(this != &rhs)
		AbstractProperty::operator=(std::move(rhs));

	return *this;
}

};	//End of namespace Property
};	//End of namespace TBTK
