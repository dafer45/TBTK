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
#include "TBTK/UnitHandler.h"

#include "TBTK/json.hpp"

using namespace std;

namespace TBTK{
namespace Property{

DOS::DOS(
	double lowerBound,
	double upperBound,
	int resolution
) :
	EnergyResolvedProperty<double>(lowerBound, upperBound, resolution)
{
}

DOS::DOS(
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
) :
	EnergyResolvedProperty(lowerBound, upperBound, resolution, data)
{
}

DOS::DOS(
	const string &serialization, Mode mode
) :
	EnergyResolvedProperty(
		Serializable::extract(
			serialization,
			mode,
			"energyResolvedProperty"
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
		break;
	default:
		TBTKExit(
			"DOS::DOS()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

string DOS::toString() const{
	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	unsigned int resolution = getResolution();

	stringstream ss;
	ss << "DOS\n";
	ss << "\tLower bound: "
		<< UnitHandler::convertNaturalToBase<Quantity::Energy>(
			lowerBound
		) << " " << UnitHandler::getUnitString<Quantity::Energy>()
		<< " (" << lowerBound << " n.u.)\n";
	ss << "\tUpper bound: "
		<< UnitHandler::convertNaturalToBase<Quantity::Energy>(
			upperBound
		) << " " << UnitHandler::getUnitString<Quantity::Energy>()
		<< " (" << upperBound << " n.u.)\n";
	ss << "\tResolution: " << resolution;

	return ss.str();
}

string DOS::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "DOS";
		j["energyResolvedProperty"] = nlohmann::json::parse(
			EnergyResolvedProperty::serialize(mode)
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
