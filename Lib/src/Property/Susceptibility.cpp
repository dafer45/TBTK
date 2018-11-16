/* Copyright 2018 Kristofer Björnson
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

/** @file Susceptibility.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/Susceptibility.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

using namespace std;

namespace TBTK{
namespace Property{

Susceptibility::Susceptibility() : EnergyResolvedProperty(){
}

Susceptibility::Susceptibility(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	unsigned int resolution
) :
	EnergyResolvedProperty(indexTree, lowerBound, upperBound, resolution)
{
}

Susceptibility::Susceptibility(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	unsigned int resolution,
	const complex<double> *data
) :
	EnergyResolvedProperty(indexTree, lowerBound, upperBound, resolution, data)
{
}

Susceptibility::Susceptibility(
	const IndexTree &indexTree,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex,
	double fundamentalMatsubaraEnergy
) :
	EnergyResolvedProperty(
		EnergyType::BosonicMatsubara,
		indexTree,
		lowerMatsubaraEnergyIndex,
		upperMatsubaraEnergyIndex,
		fundamentalMatsubaraEnergy
	)
{
}

Susceptibility::Susceptibility(
	const IndexTree &indexTree,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex,
	double fundamentalMatsubaraEnergy,
	const complex<double> *data
) :
	EnergyResolvedProperty(
		EnergyType::BosonicMatsubara,
		indexTree,
		lowerMatsubaraEnergyIndex,
		upperMatsubaraEnergyIndex,
		fundamentalMatsubaraEnergy,
		data
	)
{
}

Susceptibility::Susceptibility(
	const string &serialization,
	Mode mode
) :
	EnergyResolvedProperty(
		extract(
			serialization,
			mode,
			"energyResolvedProperty"
		),
		mode
	)
{
	TBTKAssert(
		validate(serialization, "Susceptibility", mode),
		"Property::Susceptibility::Susceptibility()",
		"Unable to parse string as Susceptibility '" << serialization
		<< "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		try{
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"Property::Susceptibility::Susceptibility()",
				"Unable to parse string as"
				<< " EnergyResolvedProperty '" << serialization
				<< "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"Property::Susceptibility::Susceptibility()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

string Susceptibility::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "Susceptibility";
		j["energyResolvedProperty"] = nlohmann::json::parse(
			EnergyResolvedProperty::serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"Property::Susceptibility::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
