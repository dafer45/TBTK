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

/** @file SelfEnergy.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/SelfEnergy.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Property{

SelfEnergy::SelfEnergy() : EnergyResolvedProperty(){
}

SelfEnergy::SelfEnergy(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	unsigned int resolution
) :
	EnergyResolvedProperty(indexTree, lowerBound, upperBound, resolution)
{
}

SelfEnergy::SelfEnergy(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	unsigned int resolution,
	const complex<double> *data
) :
	EnergyResolvedProperty(indexTree, lowerBound, upperBound, resolution, data)
{
}

SelfEnergy::SelfEnergy(
	const IndexTree &indexTree,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex,
	double fundamentalMatsubaraEnergy
) :
	EnergyResolvedProperty(
		EnergyType::FermionicMatsubara,
		indexTree,
		lowerMatsubaraEnergyIndex,
		upperMatsubaraEnergyIndex,
		fundamentalMatsubaraEnergy
	)
{
}

SelfEnergy::SelfEnergy(
	const IndexTree &indexTree,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex,
	double fundamentalMatsubaraEnergy,
	const complex<double> *data
) :
	EnergyResolvedProperty(
		EnergyType::FermionicMatsubara,
		indexTree,
		lowerMatsubaraEnergyIndex,
		upperMatsubaraEnergyIndex,
		fundamentalMatsubaraEnergy,
		data
	)
{
}

SelfEnergy::SelfEnergy(
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
		Serializable::validate(serialization, "SelfEnergy", mode),
		"Property::SelfEnergy::SelfEnergy()",
		"Unable to parse string as SelfEnergy '" << serialization
		<< "'.",
		""
	);
}

string SelfEnergy::serialize(Serializable::Mode mode) const{
	switch(mode){
	case Serializable::Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "SelfEnergy";
		j["energyResolvedProperty"] = nlohmann::json::parse(
			EnergyResolvedProperty::serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"Property::SelfEnergy::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
