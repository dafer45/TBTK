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

/** @file GreensFunction.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/GreensFunction.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/UnitHandler.h"

using namespace std;

namespace TBTK{
namespace Property{

GreensFunction::GreensFunction() : EnergyResolvedProperty(){
}

GreensFunction::GreensFunction(
	const IndexTree &indexTree,
	Type type,
	double lowerBound,
	double upperBound,
	unsigned int resolution
) :
	EnergyResolvedProperty(indexTree, lowerBound, upperBound, resolution),
	type(type)
{
}

GreensFunction::GreensFunction(
	const IndexTree &indexTree,
	Type type,
	double lowerBound,
	double upperBound,
	unsigned int resolution,
	const complex<double> *data
) :
	EnergyResolvedProperty(
		indexTree,
		lowerBound,
		upperBound,
		resolution,
		data
	),
	type(type)
{
	TBTKAssert(
		type != Type::Matsubara,
		"GreensFunction::GreensFunction()",
		"This constructor does not allow for the construction of a"
		<< " GreensFunction of type Type::Matsubara.",
		""
	);
}

GreensFunction::GreensFunction(
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
	),
	type(Type::Matsubara)
{
}

GreensFunction::GreensFunction(
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
	),
	type(Type::Matsubara)
{
}

string GreensFunction::toString() const{
	stringstream stream;
	stream << "GreensFunction\n";

	switch(getEnergyType()){
	case EnergyType::Real:
	{
		double lowerBound = getLowerBound();
		double upperBound = getUpperBound();
		unsigned int resolution = getResolution();

		stream << "\tEnergyType: Real\n";
		stream << "\tLower bound: "
			<< UnitHandler::convertNaturalToBase<Quantity::Energy>(
				lowerBound
			) << " "
			<< UnitHandler::getUnitString<Quantity::Energy>()
			<< " (" << lowerBound << " n.u.)\n";
		stream << "\tUpper bound: "
			<< UnitHandler::convertNaturalToBase<Quantity::Energy>(
				upperBound
			) << " "
			<< UnitHandler::getUnitString<Quantity::Energy>()
			<< " (" << upperBound << " n.u.)\n";
		stream << "\tResolution: " << resolution;

		break;
	}
	case EnergyType::FermionicMatsubara:
		stream << "\tEnergyType: Fermionic Matsubara";

		break;
	case EnergyType::BosonicMatsubara:
		stream << "\tEnergyType: Bosonic Matsubara";
		break;
	default:
		TBTKExit(
			"GreensFunction::toString()",
			"Unknown energy type.",
			"This should never happen, contact the developer."
		);
	}

	return stream.str();
}

};	//End of namespace Property
};	//End of namespace TBTK
