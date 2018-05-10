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

/** @file MatsubaraSusceptibility.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Functions.h"
#include "TBTK/Solver/MatsubaraSusceptibility.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

namespace TBTK{
namespace Solver{

MatsubaraSusceptibility::MatsubaraSusceptibility(
	const MomentumSpaceContext &momentumSpaceContext,
	const Property::GreensFunction2 &greensFunction
) :
	Susceptibility(Algorithm::Lindhard, momentumSpaceContext),
	greensFunction(greensFunction)
{
}

/*MatsubaraSusceptibility::MatsubaraSusceptibility(
	const MomentumSpaceContext &momentumSpaceContext,
	int *kPlusQLookupTable,
	double *fermiDiracLookupTable
) :
	Susceptibility(
		Algorithm::Lindhard,
		momentumSpaceContext,
		kPlusQLookupTable
	)
{
	this->fermiDiracLookupTable = fermiDiracLookupTable;
}

MatsubaraSusceptibility::~LindhardSusceptibility(){
	if(getIsMaster() && fermiDiracLookupTable != nullptr)
		delete [] fermiDiracLookupTable;
}*/

MatsubaraSusceptibility* MatsubaraSusceptibility::createSlave(){
	TBTKExit(
		"Solver::MatsubaraSusceptibility::createSlave()",
		"This function is not supported by this solver.",
		""
	);
}

vector<complex<double>> MatsubaraSusceptibility::calculateSusceptibility(
	const Index &index
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 5,
		"Solver::MatsubaraSusceptibility::calculateSusceptibility()",
		"The Index must be a compund Index with 5 component Indices,"
		<< "but '" << components.size() << "' components suplied.",
		""
	);
/*	const Index &kIndex = components[0];
	const Index intraBlockIndices[4] = {
		components[1],
		components[2],
		components[3],
		components[4]
	};*/

	TBTKAssert(
		greensFunction.getEnergyType()
			== Property::EnergyResolvedProperty<
				complex<double>
			>::EnergyType::FermionicMatsubara,
		"Solver::MatsubaraSusceptibility::calculateSusceptibility()",
		"The Green's function must be of the energy type"
		<< " Property::EnergyResolvedProperty::EnergyType::FermionMatsubara",
		""
	);
	unsigned int numMatsubaraEnergies
		= greensFunction.getNumMatsubaraEnergies();

	vector<complex<double>> susceptibility;
	susceptibility.reserve(numMatsubaraEnergies);
	for(unsigned int n = 0; n < numMatsubaraEnergies; n++){
		susceptibility.push_back(0.);
	}

	for(unsigned int n = 0; n < numMatsubaraEnergies; n++){
	}

	return susceptibility;
}

}	//End of namespace Solver
}	//End of namesapce TBTK
