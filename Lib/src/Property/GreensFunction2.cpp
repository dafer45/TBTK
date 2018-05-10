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

/** @file GreensFunction2.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/GreensFunction2.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Property{

/*SelfEnergy::SelfEnergy() : EnergyResolvedProperty(){
}*/

GreensFunction2::GreensFunction2(
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

GreensFunction2::GreensFunction2(
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

GreensFunction2::GreensFunction2(
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
	),
	type(Type::Matsubara)
{
}

GreensFunction2::GreensFunction2(
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
	),
	type(Type::Matsubara)
{
}

};	//End of namespace Property
};	//End of namespace TBTK
