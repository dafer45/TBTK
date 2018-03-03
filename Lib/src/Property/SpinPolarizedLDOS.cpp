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

/** @file SpinPolarizedLDOS.h
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/SpinPolarizedLDOS.h"

#include "TBTK/json.hpp"

using namespace std;
using namespace nlohmann;

namespace TBTK{
namespace Property{

SpinPolarizedLDOS::SpinPolarizedLDOS(
	int dimensions,
	const int *ranges,
	double lowerBound,
	double upperBound,
	int resolution
) :
	AbstractProperty(dimensions, ranges, resolution)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

SpinPolarizedLDOS::SpinPolarizedLDOS(
	int dimensions,
	const int *ranges,
	double lowerBound,
	double upperBound,
	int resolution,
	const SpinMatrix *data
) :
	AbstractProperty(dimensions, ranges, resolution, data)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

SpinPolarizedLDOS::SpinPolarizedLDOS(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	int resolution,
	const SpinMatrix *data
) :
	AbstractProperty(indexTree, resolution, data)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

SpinPolarizedLDOS::SpinPolarizedLDOS(
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

SpinPolarizedLDOS::SpinPolarizedLDOS(
	const SpinPolarizedLDOS &spinPolarizedLDOS
) :
	AbstractProperty(spinPolarizedLDOS)
{
	lowerBound = spinPolarizedLDOS.lowerBound;
	upperBound = spinPolarizedLDOS.upperBound;
	resolution = spinPolarizedLDOS.resolution;
}

SpinPolarizedLDOS::SpinPolarizedLDOS(
	SpinPolarizedLDOS &&spinPolarizedLDOS
) :
	AbstractProperty(std::move(spinPolarizedLDOS))
{
	lowerBound = spinPolarizedLDOS.lowerBound;
	upperBound = spinPolarizedLDOS.upperBound;
	resolution = spinPolarizedLDOS.resolution;
}

SpinPolarizedLDOS::SpinPolarizedLDOS(
	const string &serialization,
	Mode mode
) :
	AbstractProperty(
		Serializeable::extract(
			serialization,
			mode,
			"abstractProperty"
		),
		mode
	)
{
	TBTKAssert(
		validate(serialization, "SpinPolarizedLDOS", mode),
		"SpinPolarizedLDOS::SpinPolarizedLDOS()",
		"Unable to parse string as SpinPolarizedLDOS '"
		<< serialization << "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		try{
			json j = json::parse(serialization);
			lowerBound = j.at("lowerBound").get<double>();
			upperBound = j.at("upperBound").get<double>();
			resolution = j.at("resolution").get<int>();
		}
		catch(json::exception e){
			TBTKExit(
				"SpinPolarizedLDOS::SpinPolarizedLDOS()",
				"Unable to parse the string as"
				" SpinPolarizedLDOS '" << serialization
				<< "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"SpinPolarizedLDOS::SpinPolarizedLDOS()",
			"Only Serializeable::Mode::JSON is supported yet.",
			""
		);
	}
}

SpinPolarizedLDOS::~SpinPolarizedLDOS(){
}

SpinPolarizedLDOS& SpinPolarizedLDOS::operator=(const SpinPolarizedLDOS &rhs){
	if(this != &rhs){
		AbstractProperty::operator=(rhs);

		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;
	}

	return *this;
}

SpinPolarizedLDOS& SpinPolarizedLDOS::operator=(SpinPolarizedLDOS &&rhs){
	if(this != &rhs){
		AbstractProperty::operator=(std::move(rhs));

		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;
	}

	return *this;
}

string SpinPolarizedLDOS::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		json j;
		j["id"] = "SpinPolarizedLDOS";
		j["lowerBound"] = lowerBound;
		j["upperBound"] = upperBound;
		j["resolution"] = resolution;
		j["abstractProperty"] = json::parse(
			AbstractProperty::serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"SpinPolarizedLDOS::serialize()",
			"Only Serializeable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
