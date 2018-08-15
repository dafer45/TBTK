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
	const Property::GreensFunction &greensFunction
) :
	Susceptibility(Algorithm::Lindhard, momentumSpaceContext),
	greensFunction(greensFunction)
{
}

MatsubaraSusceptibility* MatsubaraSusceptibility::createSlave(){
	TBTKExit(
		"Solver::MatsubaraSusceptibility::createSlave()",
		"This function is not supported by this solver.",
		""
	);
}

vector<complex<double>> MatsubaraSusceptibility::calculateSusceptibility(
	const Index &index,
	const std::vector<std::complex<double>> &energies
){
	TBTKExit(
		"Solver::MatsubaraSusceptibility::calculateSusceptibility()",
		"This function is not supported for this solver. The"
		<< " Susceptibility cannot be calculated for arbitrary"
		<< " energies, but is restricted to a range of Matsubara"
		<< " energy indices.",
		"Use the same function with lowerMatsubaraEnergyIndex and"
		<< " upperMatsubaraEnergyIndex arguments instead of the"
		<< " energies argument. Alternatively use"
		<< " Solver::LindhardSusceptibility."
	);
}

vector<complex<double>> MatsubaraSusceptibility::calculateSusceptibility(
	const Index &index,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 5,
		"Solver::MatsubaraSusceptibility::calculateSusceptibility()",
		"The Index must be a compund Index with 5 component Indices,"
		<< "but '" << components.size() << "' components suplied.",
		""
	);
	const Index &kIndex = components[0];
	const Index intraBlockIndices[4] = {
		components[1],
		components[2],
		components[3],
		components[4]
	};

	TBTKAssert(
		lowerMatsubaraEnergyIndex <= upperMatsubaraEnergyIndex,
		"Solver::MatsubaraSusceptibility::calculateSusceptibility()",
		"The lowerMatsubaraEnergyIndex='" << lowerMatsubaraEnergyIndex
		<< "' must be less or equal to the upperMatsubaraEnergyIndex='"
		<< upperMatsubaraEnergyIndex << "'.",
		""
	);
	TBTKAssert(
		lowerMatsubaraEnergyIndex%2 == 0,
		"Solver::MatsubaraSusceptibility::calculateSusceptibility()",
		"The lowerMatsubaraEnergyIndex='" << lowerMatsubaraEnergyIndex
		<< "' must be even.",
		""
	);
	TBTKAssert(
		upperMatsubaraEnergyIndex%2 == 0,
		"Solver::MatsubaraSusceptibility::calculateSusceptibility()",
		"The upperMatsubaraEnergyIndex='" << upperMatsubaraEnergyIndex
		<< "' must be even.",
		""
	);
	unsigned int numMatsubaraEnergiesSusceptibility = (
		upperMatsubaraEnergyIndex - lowerMatsubaraEnergyIndex
	)/2 + 1;

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
	unsigned int numMatsubaraEnergiesGreensFunction
		= greensFunction.getNumMatsubaraEnergies();
	int lowerMatsubaraEnergyIndexGreensFunction
		= greensFunction.getLowerMatsubaraEnergyIndex();
	int upperMatsubaraEnergyIndexGreensFunction
		= greensFunction.getUpperMatsubaraEnergyIndex();

	vector<complex<double>> susceptibility;
	susceptibility.reserve(numMatsubaraEnergiesSusceptibility);
	for(unsigned int n = 0; n < numMatsubaraEnergiesSusceptibility; n++)
		susceptibility.push_back(0.);

	const MomentumSpaceContext &momentumSpaceContext
		= getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const vector<unsigned int> &numMeshPoints
		= momentumSpaceContext.getNumMeshPoints();
	const BrillouinZone &brillouinZone
		= momentumSpaceContext.getBrillouinZone();

	vector<unsigned int> kVector;
	kVector.reserve(kIndex.getSize());
	for(unsigned int n = 0; n < kIndex.getSize(); n++)
		kVector.push_back(kIndex[n]);
	vector<double> k = brillouinZone.getMinorMeshPoint(
		kVector,
		numMeshPoints
	);

	for(unsigned int meshPoint = 0; meshPoint < mesh.size(); meshPoint++){
		Index qIndex = brillouinZone.getMinorCellIndex(
			{mesh[meshPoint][0], mesh[meshPoint][1]},
			numMeshPoints
		);
		Index kPlusQIndex = brillouinZone.getMinorCellIndex(
			{k[0] + mesh[meshPoint][0], k[1] + mesh[meshPoint][1]},
			numMeshPoints
		);

		for(
			int susceptibilityEnergyIndex = 0;
			susceptibilityEnergyIndex
				< (int)numMatsubaraEnergiesSusceptibility;
			susceptibilityEnergyIndex++
		){
			int matsubaraEnergyIndexExternal
				= lowerMatsubaraEnergyIndex
					+ 2*susceptibilityEnergyIndex;
			for(
				int firstEnergyIndex = 0;
				firstEnergyIndex
					< (int)numMatsubaraEnergiesGreensFunction;
				firstEnergyIndex++
			){
				int matsubaraEnergyIndexInternal
					= lowerMatsubaraEnergyIndexGreensFunction
						+ 2*firstEnergyIndex;

				int  sumMatsubaraEnergyIndices
					= matsubaraEnergyIndexInternal
						+ matsubaraEnergyIndexExternal;
				if(
					sumMatsubaraEnergyIndices
					< lowerMatsubaraEnergyIndexGreensFunction
					|| sumMatsubaraEnergyIndices
					> upperMatsubaraEnergyIndexGreensFunction
				){
					continue;
				}

				int secondEnergyIndex = (
					sumMatsubaraEnergyIndices
					- lowerMatsubaraEnergyIndexGreensFunction
				)/2;

				susceptibility[susceptibilityEnergyIndex]
					-= greensFunction(
						{
							Index(
								qIndex,
								intraBlockIndices[1]
							),
							Index(
								qIndex,
								intraBlockIndices[2]
							)
						},
						firstEnergyIndex
					)*greensFunction(
						{
							Index(
								kPlusQIndex,
								intraBlockIndices[3]
							),
							Index(
								kPlusQIndex,
								intraBlockIndices[0]
							)
						},
						secondEnergyIndex
					);
			}
		}
	}

	double kT = greensFunction.getFundamentalMatsubaraEnergy()/M_PI;
	for(unsigned int n = 0; n < numMatsubaraEnergiesSusceptibility; n++)
		susceptibility[n] /= mesh.size()/kT;

	return susceptibility;
}

}	//End of namespace Solver
}	//End of namesapce TBTK
