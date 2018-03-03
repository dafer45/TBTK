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

/** @file ParametrizedLine.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/ParametrizedLine.h"

using namespace std;

namespace TBTK{

ParametrizedLine::ParametrizedLine(
	initializer_list<double> start,
	initializer_list<double> direction
){
	TBTKAssert(
		start.size() == direction.size(),
		"ParametrizedLine::ParametrizedLine()",
		"Incompatible vector dimensions. 'start' has " << start.size()
		<< " components, while 'direction' has " << direction.size()
		<< " components.",
		""
	);

	for(unsigned int n = 0; n < start.size(); n++){
		this->start.push_back(*(start.begin() + n));
		this->direction.push_back(*(direction.begin() + n));
	}
}

ParametrizedLine::ParametrizedLine(
	const std::vector<double> &start,
	const std::vector<double> &direction
){
	TBTKAssert(
		start.size() == direction.size(),
		"ParametrizedLine::ParametrizedLine()",
		"Incompatible vector dimensions. 'start' has " << start.size()
		<< " components, while 'direction' has " << direction.size()
		<< " components.",
		""
	);

	for(unsigned int n = 0; n < start.size(); n++){
		this->start.push_back(start.at(n));
		this->direction.push_back(direction.at(n));
	}
}

vector<double> ParametrizedLine::operator()(
	std::initializer_list<double> lambda
) const{
	vector<double> result;
	for(unsigned int n = 0; n < start.size(); n++)
		result.push_back(start.at(n) + (*lambda.begin())*direction.at(n));

	return result;
}

};	//End of namespace TBTK
