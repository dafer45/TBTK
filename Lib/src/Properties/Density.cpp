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

/** @file Density.cpp
 *
 *  @author Kristofer Björnson
 */

#include "Density.h"

namespace TBTK{
namespace Property{

Density::Density(
	int dimensions,
	const int *ranges
) :
	AbstractProperty(dimensions, ranges, 1)
{
}

Density::Density(
	int dimensions,
	const int *ranges,
	const double *data
) :
	AbstractProperty(dimensions, ranges, 1, data)
{
}

Density::Density(
	const Density &density
) :
	AbstractProperty(density)
{
}

Density::Density(
	Density &&density
) :
	AbstractProperty(std::move(density))
{
}

Density::~Density(){
}

Density& Density::operator=(const Density &rhs){
	AbstractProperty::operator=(rhs);

	return *this;
}

Density& Density::operator=(Density &&rhs){
	if(this != &rhs){
		AbstractProperty::operator=(std::move(rhs));
	}

	return *this;
}

};	//End of namespace Property
};	//End of namespace TBTK
