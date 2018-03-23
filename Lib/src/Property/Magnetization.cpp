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

#include "TBTK/Property/Magnetization.h"

#include "TBTK/json.hpp"

using namespace std;
using namespace nlohmann;

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
	const SpinMatrix *data
) :
	AbstractProperty(dimensions, ranges, 4, data)
{
}

Magnetization::Magnetization(
	const IndexTree &indexTree
) :
	AbstractProperty(indexTree, 4)
{
}

Magnetization::Magnetization(
	const IndexTree &indexTree,
	const SpinMatrix *data
) :
	AbstractProperty(indexTree, 4, data)
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

Magnetization::Magnetization(
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
		validate(serialization, "Magnetization", mode)
,		"Magnetization::Magnetization()",
		"Unable to parse string as Magnetization '" << serialization
		<< "'.",
		""
	);
}

Magnetization::~Magnetization(){
}

Magnetization& Magnetization::operator=(const Magnetization &rhs){
	if(this != &rhs)
		AbstractProperty::operator=(rhs);

	return *this;
}

Magnetization& Magnetization::operator=(Magnetization &&rhs){
	if(this != &rhs)
		AbstractProperty::operator=(std::move(rhs));

	return *this;
}

string Magnetization::serialize(Mode mode) const{
	switch(mode){
	case Mode ::JSON:
	{
		json j;
		j["id"] = "Magnetization";
		j["abstractProperty"] = json::parse(
			AbstractProperty::serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"Magnetization::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
