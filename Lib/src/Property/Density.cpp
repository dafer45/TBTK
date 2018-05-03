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

#include "TBTK/Property/Density.h"

#include "TBTK/json.hpp"

using namespace std;
using namespace nlohmann;

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
	const IndexTree &indexTree
) :
	AbstractProperty(indexTree, 1)
{
}

Density::Density(
	const IndexTree &indexTree,
	const double *data
) :
	AbstractProperty(indexTree, 1, data)
{
}

/*Density::Density(
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
}*/

Density::Density(
	const string &serialization,
	Mode mode
) :
	AbstractProperty(
		Serializable::extract(
			serialization,
			mode,
			"abstractProperty"
		),
		mode
	)
{
	TBTKAssert(
		validate(serialization, "Density", mode),
		"Density::Density()",
		"Unable to parse string as Density '" << serialization << "'.",
		""
	);
}

/*Density::~Density(){
}

Density& Density::operator=(const Density &rhs){
	if(this != &rhs)
		AbstractProperty::operator=(rhs);

	return *this;
}

Density& Density::operator=(Density &&rhs){
	if(this != &rhs)
		AbstractProperty::operator=(std::move(rhs));

	return *this;
}*/

double Density::getMin() const{
	const std::vector<double> &data = getData();
	double min = data[0];
	for(unsigned int n = 1; n < data.size(); n++)
		if(data[n] < min)
			min = data[n];

	return min;
}

double Density::getMax() const{
	const std::vector<double> &data = getData();
	double max = data[0];
	for(unsigned int n = 1; n < data.size(); n++)
		if(data[n] > max)
			max = data[n];

	return max;
}

string Density::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		json j;
		j["id"] = "Density";
		j["abstractProperty"] = json::parse(
			AbstractProperty::serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"Density::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
