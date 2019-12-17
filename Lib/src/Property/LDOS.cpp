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
#include "TBTK/UnitHandler.h"

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
	EnergyResolvedProperty(ranges, lowerBound, upperBound, resolution)
{
}

LDOS::LDOS(
	const std::vector<int> &ranges,
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
) :
	EnergyResolvedProperty(
		ranges,
		lowerBound,
		upperBound,
		resolution,
		data
	)
{
}

LDOS::LDOS(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	int resolution
) :
	EnergyResolvedProperty(indexTree, lowerBound, upperBound, resolution)
{
}

LDOS::LDOS(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
) :
	EnergyResolvedProperty(
		indexTree,
		lowerBound,
		upperBound,
		resolution,
		data
	)
{
}

LDOS::LDOS(
	const string &serialization,
	Mode mode
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
		validate(serialization, "LDOS", mode),
		"LDOS::LDOS()",
		"Unable to parse string as LDOS '" << serialization << "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		break;
	default:
		TBTKExit(
			"LDOS::LDOS()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

string LDOS::toString() const{
	unsigned int numSites = getSize()/getBlockSize();
	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	unsigned int resolution = getResolution();

	stringstream ss;
	ss << "LDOS\n";
	ss << "\tNumber of sites: " << numSites << "\n";
	ss << "\tLower bound: "
		<< UnitHandler::convertNaturalToBase<Quantity::Energy>(
			lowerBound
		) << " " << UnitHandler::getUnitString<Quantity::Energy>() << " ("
		<< lowerBound << " n.u.)\n";
	ss << "\tUpper bound: "
		<< UnitHandler::convertNaturalToBase<Quantity::Energy>(
			upperBound
		) << " " << UnitHandler::getUnitString<Quantity::Energy>() << " ("
		<< upperBound << " n.u.)\n";
	ss << "\tResolution: " << resolution;

	return ss.str();
}

string LDOS::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "LDOS";
		j["energyResolvedProperty"] = nlohmann::json::parse(
			EnergyResolvedProperty::serialize(mode)
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
