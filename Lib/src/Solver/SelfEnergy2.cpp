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

#include "TBTK/Array.h"
#include "TBTK/Convolver.h"
#include "TBTK/Solver/SelfEnergy2.h"
#include "TBTK/UnitHandler.h"

#include <complex>

using namespace std;

namespace TBTK{
namespace Solver{

SelfEnergy2::SelfEnergy2(
	const RPA::MomentumSpaceContext &momentumSpaceContext,
	const Property::InteractionVertex &interactionVertex,
	const Property::GreensFunction &greensFunction
) :
	Communicator(true),
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
		abs(lowerMatsubaraEnergyIndex)%2 == 1,
		"Solver::SelfEnergy2::calclulateSelfEnergy()",
		"The lowerMatsubaraEnergyIndex='" << lowerMatsubaraEnergyIndex
		<< "' must be odd.",
		""
	);
	TBTKAssert(
		abs(upperMatsubaraEnergyIndex)%2 == 1,
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
		<< " Property::EnergyResolvedProperty::EnergyType::FermionicMatsubara",
		""
	);
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
		<< " Property::EnergyResolvedProperty::EnergyType::BosonicMatsubara",
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
		Index kMinusQIndex = brillouinZone.getMinorCellIndex(
			{k[0] - mesh[meshPoint][0], k[1] - mesh[meshPoint][1]},
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
					= matsubaraEnergyIndexExternal
						- matsubaraEnergyIndexInternal;

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
									intraBlockIndices[0],
									{orbital1},
									{orbital0},
									intraBlockIndices[1]
								},
								firstEnergyIndex
						)*greensFunction(
							{
								Index(
									kMinusQIndex,
									{orbital1}
								),
								Index(
									kMinusQIndex,
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

Property::SelfEnergy SelfEnergy2::calculateSelfEnergyAllBlocks(
	const Index &index,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 2,
		"Solver::SelfEnergy2::calculateSelfEnergyAllBlocks()",
		"The Index must be a compund Index with 2 component Indices,"
		<< " but '" << components.size() << "' components suplied.",
		""
	);
	for(unsigned int n = 0; n < components.size(); n++){
		TBTKAssert(
			components[n].getSize() == 1,
			"Solver::SelfEnergy2::calculateSelfEnergyAllBlocks()",
			"Currently only intra block indices with a single"
			<< " subindex are allowed, but the Index component '"
			<< components[n].toString() << "' has '"
			<< components[n].getSize() << "' subindices.",
			""
		);
	}
	const Index intraBlockIndices[2] = {
		components[0],
		components[1]
	};

	TBTKAssert(
		lowerMatsubaraEnergyIndex <= upperMatsubaraEnergyIndex,
		"Solver::SelfEnergy2::calculateSelfEnergyAllBlocks()",
		"The lowerMatsubaraEnergyIndex='" << lowerMatsubaraEnergyIndex
		<< "' must be less or equal to the upperMatsubaraEnergyIndex='"
		<< upperMatsubaraEnergyIndex << "'.",
		""
	);
	TBTKAssert(
		abs(lowerMatsubaraEnergyIndex)%2 == 1,
		"Solver::SelfEnergy2::calclulateSelfEnergyAllBlocks()",
		"The lowerMatsubaraEnergyIndex='" << lowerMatsubaraEnergyIndex
		<< "' must be odd.",
		""
	);
	TBTKAssert(
		abs(upperMatsubaraEnergyIndex)%2 == 1,
		"Solver::SelfEnergy2::calclulateSelfEnergyAllBlocks()",
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
		"Solver::SelfEnergy2::claculateSelfEnergyAllBlocks()",
		"The Green's function must be of the energy type"
		<< " Property::EnergyResolvedProperty::EnergyType::FermionicMatsubara",
		""
	);
	unsigned int numMatsubaraEnergiesGreensFunction
		= greensFunction.getNumMatsubaraEnergies();
	int lowerMatsubaraEnergyIndexGreensFunction
		= greensFunction.getLowerMatsubaraEnergyIndex();
	int upperMatsubaraEnergyIndexGreensFunction
		= greensFunction.getUpperMatsubaraEnergyIndex();

	TBTKAssert(
		interactionVertex.getEnergyType()
			== Property::EnergyResolvedProperty<
				complex<double>
			>::EnergyType::BosonicMatsubara,
		"Solver::SelfEnergy2::calculateSelfEnergyAllBlocks()",
		"The InteractionVertex function must be of the energy type"
		<< " Property::EnergyResolvedProperty::EnergyType::BosonicMatsubara",
		""
	);
	unsigned int numMatsubaraEnergiesInteractionVertex
		= interactionVertex.getNumMatsubaraEnergies();
	int lowerMatsubaraEnergyIndexInteractionVertex
		= interactionVertex.getLowerMatsubaraEnergyIndex();
	int upperMatsubaraEnergyIndexInteractionVertex
		= interactionVertex.getLowerMatsubaraEnergyIndex();

	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const vector<unsigned int> &numMeshPoints
		= momentumSpaceContext.getNumMeshPoints();
	const BrillouinZone &brillouinZone
		= momentumSpaceContext.getBrillouinZone();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	TBTKAssert(
		numMeshPoints.size() == 2,
		"Solver::SelfEnergy2::calculateSelfEnergyAllBlocks()",
		"Only two-dimensional momentum spaces supported yet, but the"
		<< " MomentumSpaceContext describes a '"
		<< numMeshPoints.size() << "'-dimensional momentum space.",
		""
	);

	int minMatsubaraEnergyIndex = min(
		lowerMatsubaraEnergyIndexInteractionVertex,
		lowerMatsubaraEnergyIndexGreensFunction
	);
	int maxMatsubaraEnergyIndex = max(
		upperMatsubaraEnergyIndexInteractionVertex,
		upperMatsubaraEnergyIndexGreensFunction
	);
	int numMatsubaraEnergiesCrossCorrelation = 2*(
		(maxMatsubaraEnergyIndex - minMatsubaraEnergyIndex)/2 + 1
	);
	vector<unsigned int> crossCorrelationRanges;
	for(unsigned int n = 0; n < numMeshPoints.size(); n++)
		crossCorrelationRanges.push_back(numMeshPoints[n]);
	crossCorrelationRanges.push_back(numMatsubaraEnergiesCrossCorrelation);

	Array<complex<double>> selfEnergyArray;
	for(unsigned int orbital0 = 0; orbital0 < numOrbitals; orbital0++){
		for(
			unsigned int orbital1 = 0;
			orbital1 < numOrbitals;
			orbital1++
		){
			Array<complex<double>> interactionVertexArray(
				crossCorrelationRanges,
				0
			);
			Array<complex<double>> greensFunctionArray(
				crossCorrelationRanges,
				0
			);

			for(
				unsigned int meshPoint = 0;
				meshPoint < mesh.size();
				meshPoint++
			){
				Index qIndex = brillouinZone.getMinorCellIndex(
					{mesh[meshPoint][0], mesh[meshPoint][1]},
					numMeshPoints
				);

				for(
					unsigned int n = 0;
					n < numMatsubaraEnergiesInteractionVertex;
					n++
				){
					int energyIndex = (
						lowerMatsubaraEnergyIndexInteractionVertex
						+ 2*(int)n
					)/2;
					while(energyIndex < 0){
						energyIndex
							+= numMatsubaraEnergiesCrossCorrelation;
					}
					interactionVertexArray[{
						(unsigned int)qIndex[0],
						(unsigned int)qIndex[1],
						(unsigned int)energyIndex
					}] = interactionVertex(
						{
							qIndex,
							intraBlockIndices[0],
							{orbital1},
							{orbital0},
							intraBlockIndices[1]
						},
						n
					);
				}

				for(
					unsigned int n = 0;
					n < numMatsubaraEnergiesGreensFunction;
					n++
				){
					int energyIndex = (
						lowerMatsubaraEnergyIndexGreensFunction
						+ 2*(int)n - 1
					)/2;
					while(energyIndex < 0){
						energyIndex
							+= numMatsubaraEnergiesCrossCorrelation;
					}
					greensFunctionArray[{
						(unsigned int)qIndex[0],
						(unsigned int)qIndex[1],
						(unsigned int)energyIndex
					}] = greensFunction(
						{
							Index(
								qIndex,
								{orbital1}
							),
							Index(
								qIndex,
								{orbital0}
							)
						},
						n
					);
				}
			}

			if(orbital0 == 0 && orbital1 == 0){
				selfEnergyArray = Convolver::convolve(
					interactionVertexArray,
					greensFunctionArray
				);
			}
			else{
				selfEnergyArray
					= selfEnergyArray
						+ Convolver::convolve(
							interactionVertexArray,
							greensFunctionArray
						);
			}
		}
	}

	double kT = greensFunction.getFundamentalMatsubaraEnergy()/M_PI;
	for(unsigned int meshPoint = 0; meshPoint < mesh.size(); meshPoint++){
		for(
			int n = 0;
			n < numMatsubaraEnergiesCrossCorrelation;
			n++
		){
			selfEnergyArray[
				numMatsubaraEnergiesCrossCorrelation*meshPoint
				+ n
			] /= mesh.size()/kT;
		}
	}

	IndexTree memoryLayout;
	for(unsigned int kx = 0; kx < numMeshPoints[0]; kx++){
		for(unsigned int ky = 0; ky < numMeshPoints[1]; ky++){
			memoryLayout.add({
				{kx, ky},
				intraBlockIndices[0],
				intraBlockIndices[1]
			});
		}
	}
	memoryLayout.generateLinearMap();

	Property::SelfEnergy selfEnergy(
		memoryLayout,
		lowerMatsubaraEnergyIndex,
		upperMatsubaraEnergyIndex,
		greensFunction.getFundamentalMatsubaraEnergy()
	);

	for(unsigned int kx = 0; kx < numMeshPoints[0]; kx++){
		for(unsigned int ky = 0; ky < numMeshPoints[1]; ky++){
			for(
				unsigned int n = 0;
				n < numMatsubaraEnergiesSelfEnergy;
				n++
			){
				int energyIndex = (
					lowerMatsubaraEnergyIndex + 2*(int)n - 1
				)/2;

				if(
					energyIndex
						< -(int)numMatsubaraEnergiesCrossCorrelation/2
					|| energyIndex
						> (int)numMatsubaraEnergiesCrossCorrelation/2
				){
					selfEnergy(
						{
							{kx, ky},
							intraBlockIndices[0],
							intraBlockIndices[1]
						},
						n
					) = 0;
				}
				else{
					if(energyIndex < 0)
						energyIndex += 2*(int)numMatsubaraEnergiesGreensFunction;

					selfEnergy(
						{
							{kx, ky},
							intraBlockIndices[0],
							intraBlockIndices[1]
						},
						n
					) = selfEnergyArray[
						numMatsubaraEnergiesCrossCorrelation*(
							numMeshPoints[1]*kx
							+ ky
						) + energyIndex
					];
				}
			}
		}
	}

	return selfEnergy;
}

}	//End of namespace Solver
}	//End of namesapce TBTK
