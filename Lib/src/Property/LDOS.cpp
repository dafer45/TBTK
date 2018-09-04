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

/** @file LDOS.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/LDOS.h"

#include "TBTK/json.hpp"

using namespace std;

namespace TBTK{
namespace Property{

LDOS::LDOS(
	const std::vector<int> &ranges,
	double lowerBound,
	double upperBound,
	int resolution
) :
	AbstractProperty(ranges, resolution)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

LDOS::LDOS(
	const std::vector<int> &ranges,
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
) :
	AbstractProperty(ranges, resolution, data)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

LDOS::LDOS(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	int resolution
) :
	AbstractProperty(indexTree, resolution)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

LDOS::LDOS(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
) :
	AbstractProperty(indexTree, resolution, data)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

LDOS::LDOS(
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
		validate(serialization, "LDOS", mode),
		"LDOS::LDOS()",
		"Unable to parse string as LDOS '" << serialization << "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			lowerBound = j.at("lowerBound").get<double>();
			upperBound = j.at("upperBound").get<double>();
			resolution = j.at("resolution").get<int>();
		}
		catch(nlohmann::json::exception e){
			TBTKExit(
				"LDOS::LDOS()",
				"Unable to parse string as LDOS '"
				<< serialization << "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"LDOS::LDOS()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

string LDOS::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "LDOS";
		j["lowerBound"] = lowerBound;
		j["upperBound"] = upperBound;
		j["resolution"] = resolution;
		j["abstractProperty"] = nlohmann::json::parse(
			AbstractProperty::serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"LDOS::serialize()",
			"Onle Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
