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

/** @Geometry.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Geometry.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

using namespace std;
//using namespace nlohmann;

namespace TBTK{

Geometry::Geometry(
){
	dimensions = -1;
}

Geometry::Geometry(
	const string &serialization,
	Mode mode
){
	TBTKAssert(
		validate(serialization, "Geometry", mode),
		"Geometry::Geometry()",
		"Unable to parse string as Geometry '" << serialization
		<< "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
	{
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			dimensions = j.at("dimensions").get<int>();
			coordinates = IndexedDataTree<vector<double>>(
				j.at("coordinates"),
				mode
			);
		}
		catch(nlohmann::json::exception e){
			TBTKExit(
				"Geometry::Geometry()",
				"Unable to parse string as Geometry '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"Geometry::Geometry()",
			"Only Serializable::Mode:Debug is supported yet.",
			""
		);
	}
}

Geometry::~Geometry(
){
}

void Geometry::setCoordinates(
	const Index &index,
	std::initializer_list<double> coordinates
){
	this->coordinates.add(coordinates, index);
}

void Geometry::setCoordinates(
	const Index &index,
	const std::vector<double> &coordinates
){
	this->coordinates.add(coordinates, index);
}

void Geometry::translate(const vector<double> &translation){
	TBTKNotYetImplemented("Geometry::translate()");
}

string Geometry::serialize(Mode mode) const{
	TBTKNotYetImplemented("Geometry::serialize()");
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "Geometry";
		j["dimensions"] = dimensions;
		j["coordinates"] = nlohmann::json::parse(
			coordinates.serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"Geometry::Geometry()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
