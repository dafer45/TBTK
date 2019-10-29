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

#include "TBTK/Property/EigenValues.h"

#include "TBTK/json.hpp"

using namespace std;

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
		validate(serialization, "EigenValues", mode),
		"EigenValues::EigenValues()",
		"Unable to parse string as EigenValues '" << serialization
		<< "'.",
		""
	);
}

string EigenValues::toString() const{
	stringstream stream;
	stream << "EigenValues\n";
	stream << "\tNumber of eigen values: " << getBlockSize();

	return stream.str();
}

string EigenValues::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "EigenValues";
		j["abstractProperty"] = nlohmann::json::parse(
			AbstractProperty::serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"EigenValues::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
