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

/** @file SelfEnergy2.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Solver/SelfEnergy2.h"
#include "TBTK/UnitHandler.h"

#include <complex>

using namespace std;

namespace TBTK{
namespace Solver{

SelfEnergy2::SelfEnergy2(
	const MomentumSpaceContext &momentumSpaceContext,
	const Property::InteractionVertex &interactionVertex,
	const Property::GreensFunction &greensFunction
) :
	momentumSpaceContext(momentumSpaceContext),
	interactionVertex(interactionVertex),
	greensFunction(greensFunction)
{
}

vector<complex<double>> SelfEnergy2::calculateSelfEnergy(
	const Index &index,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 3,
		"Solver::SelfEnergy2::calculateSelfEnergy()",
		"The Index must be a compund Index with 3 component Indices,"
		<< " but '" << components.size() << "' components suplied.",
		""
	);
	for(unsigned int n = 1; n < components.size(); n++){
		TBTKAssert(
			components[n].getSize() == 1,
			"Solver::SelfEnergy2::calculateSelfEnergy()",
			"Currently only intra block indices with a single"
			<< " subindex are allowed, but the Index component '"
			<< components[n].toString() << "' has '"
			<< components[n].getSize() << "' subindices.",
			""
		);
	}
	const Index &kIndex = components[0];
	const Index intraBlockIndices[2] = {
		components[1],
		components[2]
	};

	TBTKAssert(
		lowerMatsubaraEnergyIndex <= upperMatsubaraEnergyIndex,
		"Solver::SelfEnergy2::calculateSelfEnergy()",
		"The lowerMatsubaraEnergyIndex='" << lowerMatsubaraEnergyIndex
		<< "' must be less or equal to the upperMatsubaraEnergyIndex='"
		<< upperMatsubaraEnergyIndex << "'.",
		""
	);
	TBTKAssert(
		lowerMatsubaraEnergyIndex%2 == 1,
		"Solver::SelfEnergy2::calclulateSelfEnergy()",
		"The lowerMatsubaraEnergyIndex='" << lowerMatsubaraEnergyIndex
		<< "' must be odd.",
		""
	);
	TBTKAssert(
		upperMatsubaraEnergyIndex%2 == 1,
		"Solver::SelfEnergy2::calclulateSelfEnergy()",
		"The upperMatsubaraEnergyIndex='" << upperMatsubaraEnergyIndex
		<< "' must be odd.",
		""
	);
	unsigned int numMatsubaraEnergiesSelfEnergy = (
		upperMatsubaraEnergyIndex - lowerMatsubaraEnergyIndex
	)/ 2 + 1;

	TBTKAssert(
		greensFunction.getEnergyType()
			== Property::EnergyResolvedProperty<
				complex<double>
			>::EnergyType::FermionicMatsubara,
		"Solver::SelfEnergy2::claculateSelfEnergy()",
		"The Green's function must be of the energy type"
		<< " Property::EnergyResolvedProperty::EnergyType::FermionMatsubara",
		""
	);
//	unsigned int numMatsubaraEnergiesGreensFunction
//		= greensFunction.getNumMatsubaraEnergies();
	int lowerMatsubaraEnergyIndexGreensFunction
		= greensFunction.getLowerMatsubaraEnergyIndex();
	int upperMatsubaraEnergyIndexGreensFunction
		= greensFunction.getUpperMatsubaraEnergyIndex();

	TBTKAssert(
		interactionVertex.getEnergyType()
			== Property::EnergyResolvedProperty<
				complex<double>
			>::EnergyType::BosonicMatsubara,
		"Solver::SelfEnergy2::calculateSelfEnergy()",
		"The InteractionVertex function must be of the energy type"
		<< " Property::EnergyResolvedProperty::EnergyType::FermionMatsubara",
		""
	);
	unsigned int numMatsubaraEnergiesInteractionVertex
		= interactionVertex.getNumMatsubaraEnergies();
	int lowerMatsubaraEnergyIndexInteractionVertex
		= interactionVertex.getLowerMatsubaraEnergyIndex();

	vector<complex<double>> selfEnergy;
	selfEnergy.reserve(numMatsubaraEnergiesSelfEnergy);
	for(unsigned int n = 0; n < numMatsubaraEnergiesSelfEnergy; n++)
		selfEnergy.push_back(0.);

	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const vector<unsigned int> &numMeshPoints
		= momentumSpaceContext.getNumMeshPoints();
	const BrillouinZone &brillouinZone
		= momentumSpaceContext.getBrillouinZone();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

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
		Index qMinusKIndex = brillouinZone.getMinorCellIndex(
			{mesh[meshPoint][0] - k[0], mesh[meshPoint][1] - k[1]},
			numMeshPoints
		);

		for(
			int selfEnergyIndex = 0;
			selfEnergyIndex < (int)numMatsubaraEnergiesSelfEnergy;
			selfEnergyIndex++
		){
			int matsubaraEnergyIndexExternal
				= lowerMatsubaraEnergyIndex
					+ 2*selfEnergyIndex;
			for(
				int firstEnergyIndex = 0;
				firstEnergyIndex < (int)numMatsubaraEnergiesInteractionVertex;
				firstEnergyIndex++
			){
				int matsubaraEnergyIndexInternal
					= lowerMatsubaraEnergyIndexInteractionVertex
						+ 2*firstEnergyIndex;

				int differenceMatsubaraEnergyIndices
					= matsubaraEnergyIndexInternal
						- matsubaraEnergyIndexExternal;

				if(
					differenceMatsubaraEnergyIndices
					< lowerMatsubaraEnergyIndexGreensFunction
					|| differenceMatsubaraEnergyIndices
					> upperMatsubaraEnergyIndexGreensFunction
				){
					continue;
				}

				int secondEnergyIndex = (
					differenceMatsubaraEnergyIndices
					- lowerMatsubaraEnergyIndexGreensFunction
				)/2;

				for(
					unsigned int orbital0 = 0;
					orbital0 < numOrbitals;
					orbital0++
				){
					for(
						unsigned int orbital1 = 0;
						orbital1 < numOrbitals;
						orbital1++
					){
						selfEnergy[selfEnergyIndex]
							+= interactionVertex(
								{
									qIndex,
									{orbital0},
									intraBlockIndices[1],
									intraBlockIndices[0],
									{orbital1}
								},
								firstEnergyIndex
						)*greensFunction(
							{
								Index(
									qMinusKIndex,
									{orbital1}
								),
								Index(
									qMinusKIndex,
									{orbital0}
								)
							},
							secondEnergyIndex
						);
					}
				}
			}
		}
	}

	double kT = greensFunction.getFundamentalMatsubaraEnergy()/M_PI;
	for(unsigned int n = 0; n < numMatsubaraEnergiesSelfEnergy; n++)
		selfEnergy[n] /= mesh.size()/kT;

	return selfEnergy;
}

}	//End of namespace Solver
}	//End of namesapce TBTK
