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

/** @file Range.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Range.h"

#include "TBTK/json.hpp"

using namespace std;
//using namespace nlohmann;

namespace TBTK{

Range::Range(
	double lowerBound,
	double upperBound,
	unsigned int resolution,
	bool includeLowerBound,
	bool includeUpperBound
){
	TBTKAssert(
		resolution > 1,
		"Range::Range()",
		"'resolution' must be larger than 1.",
		""
	);

	if(includeLowerBound && includeUpperBound)
		dx = (upperBound - lowerBound)/(resolution - 1);
	else if(includeLowerBound || includeUpperBound)
		dx = (upperBound - lowerBound)/resolution;
	else
		dx = (upperBound - lowerBound)/(resolution+1);

	if(includeLowerBound)
		start = lowerBound;
	else
		start = lowerBound + dx;

	this->resolution = resolution;
}

Range::Range(const string &serialization, Mode mode){
	TBTKAssert(
		validate(serialization, "Range", mode),
		"Range::Range()",
		"Unable to parse string as Range '" << serialization << "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			start = j.at("start").get<double>();
			dx = j.at("dx").get<double>();
			resolution = j.at("resolution").get<unsigned int>();
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"Range::Range()",
				"Unable to parse string as Range '"
				<< serialization << "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"Range::Range()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

std::string Range::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "Range";
		j["start"] = start;
		j["dx"] = dx;
		j["resolution"] = resolution;

		return j.dump();
	}
	default:
		TBTKExit(
			"Range::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
