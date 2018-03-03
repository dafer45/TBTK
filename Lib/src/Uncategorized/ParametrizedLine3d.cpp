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

/** @file ParametrizedLine3d.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/ParametrizedLine3d.h"

using namespace std;

namespace TBTK{

ParametrizedLine3d::ParametrizedLine3d(
	const Vector3d &start,
	const Vector3d &direction
){
	this->start = start;
	this->direction = direction;
}

Vector3d ParametrizedLine3d::operator()(
	std::initializer_list<double> lambda
) const{
	return start + (*lambda.begin())*direction;
}

};	//End of namespace TBTK
