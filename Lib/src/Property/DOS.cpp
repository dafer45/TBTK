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

/** @file DOS.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/DOS.h"
#include "TBTK/Streams.h"

#include "TBTK/json.hpp"

using namespace std;

namespace TBTK{
namespace Property{

DOS::DOS(
	double lowerBound,
	double upperBound,
	int resolution
) :
	AbstractProperty(resolution)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

DOS::DOS(
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
) :
	AbstractProperty(resolution, data)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

DOS::DOS(
	const string &serialization, Mode mode
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
		validate(serialization, "DOS", mode),
		"DOS::DOS()",
		"Unable to parse string as DOS '" << serialization << "'.",
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
				"DOS::DOS()",
				"Unable to parse string as DOS '"
				<< serialization << "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"DOS::DOS()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

string DOS::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "DOS";
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
			"DOS::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
