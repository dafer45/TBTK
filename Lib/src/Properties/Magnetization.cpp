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

/** @file Magnetization.cpp
 *
 *  @author Kristofer Björnson
 */

#include "Magnetization.h"

using namespace std;

namespace TBTK{
namespace Property{

Magnetization::Magnetization(
	int dimensions,
	const int* ranges
) :
	AbstractProperty(dimensions, ranges, 4)
{
}

Magnetization::Magnetization(
	int dimensions,
	const int* ranges,
	const complex<double> *data
) :
	AbstractProperty(dimensions, ranges, 4, data)
{
}

Magnetization::Magnetization(
	const Magnetization &magnetization
) :
	AbstractProperty(magnetization)
{
}

Magnetization::Magnetization(
	Magnetization &&magnetization
) :
	AbstractProperty(std::move(magnetization))
{
}

Magnetization::~Magnetization(){
}

Magnetization& Magnetization::operator=(const Magnetization &rhs){
	AbstractProperty::operator=(rhs);

	return *this;
}

Magnetization& Magnetization::operator=(Magnetization &&rhs){
	if(this != &rhs)
		AbstractProperty::operator=(std::move(rhs));

	return *this;
}

};	//End of namespace Property
};	//End of namespace TBTK
