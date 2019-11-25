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

#include "TBTK/Array.h"
#include "TBTK/Convolver.h"
#include "TBTK/FourierTransform.h"
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
	Communicator(true),
	greensFunction(greensFunction),
	momentumSpaceContext(momentumSpaceContext)
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
		<< " but '" << components.size() << "' components suplied.",
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
								intraBlockIndices[3]
							),
							Index(
								qIndex,
								intraBlockIndices[0]
							)
						},
						firstEnergyIndex
					)*greensFunction(
						{
							Index(
								kPlusQIndex,
								intraBlockIndices[1]
							),
							Index(
								kPlusQIndex,
								intraBlockIndices[2]
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

Property::Susceptibility MatsubaraSusceptibility::calculateSusceptibilityAllBlocks(
	const Index &index,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 4,
		"Solver::MatsubaraSusceptibility::calculateSusceptibility()",
		"The Index must be a compund Index with 4 component Indices,"
		<< "but '" << components.size() << "' components suplied.",
		""
	);
	const Index intraBlockIndices[4] = {
		components[0],
		components[1],
		components[2],
		components[3]
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

	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const vector<unsigned int> &numMeshPoints
		= momentumSpaceContext.getNumMeshPoints();
	const BrillouinZone &brillouinZone
		= momentumSpaceContext.getBrillouinZone();

	TBTKAssert(
		numMeshPoints.size() == 2,
		"Solver::MatsubaraSusceptibility::calculateSusceptibility()",
		"Only two-dimensional momentum spaces supported yet, but the"
		<< " MomentumSpaceContext describes a '"
		<< numMeshPoints.size() << "'-dimensional momentum space.",
		""
	);

	unsigned int numMatsubaraEnergiesGreensFunction
		= greensFunction.getNumMatsubaraEnergies();
	vector<unsigned int> crossCorrelationRanges;
	for(unsigned int n = 0; n < numMeshPoints.size(); n++)
		crossCorrelationRanges.push_back(numMeshPoints[n]);
	crossCorrelationRanges.push_back(2*numMatsubaraEnergiesGreensFunction);

	Array<complex<double>> greensFunction0In
		= Array<complex<double>>::create(crossCorrelationRanges, 0);
	Array<complex<double>> greensFunction1In
		= Array<complex<double>>::create(crossCorrelationRanges, 0);

	#pragma omp parallel for
	for(unsigned int meshPoint = 0; meshPoint < mesh.size(); meshPoint++){
		Index qIndex = brillouinZone.getMinorCellIndex(
			{mesh[meshPoint][0], mesh[meshPoint][1]},
			numMeshPoints
		);

		for(
			unsigned int n = 0;
			n < numMatsubaraEnergiesGreensFunction;
			n++
		){
			greensFunction0In[{
				(unsigned int)qIndex[0],
				(unsigned int)qIndex[1],
				n
			}] = conj(greensFunction(
				{
					Index(qIndex, intraBlockIndices[3]),
					Index(qIndex, intraBlockIndices[0])
				},
				n
			));
			greensFunction1In[{
				(unsigned int)qIndex[0],
				(unsigned int)qIndex[1],
				n
			}] = greensFunction(
				{
					Index(qIndex, intraBlockIndices[1]),
					Index(qIndex, intraBlockIndices[2])
				},
				n
			);
		}
	}

	Array<complex<double>> susceptibilityOut = Convolver::crossCorrelate(
		greensFunction0In,
		greensFunction1In
	);

	double kT = greensFunction.getFundamentalMatsubaraEnergy()/M_PI;
	for(unsigned int meshPoint = 0; meshPoint < mesh.size(); meshPoint++){
		for(unsigned int n = 0; n < 2*numMatsubaraEnergiesGreensFunction; n++){
			susceptibilityOut[
				2*numMatsubaraEnergiesGreensFunction*meshPoint
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
				intraBlockIndices[1],
				intraBlockIndices[2],
				intraBlockIndices[3]
			});
		}
	}
	memoryLayout.generateLinearMap();

	Property::Susceptibility susceptibility(
		memoryLayout,
		lowerMatsubaraEnergyIndex,
		upperMatsubaraEnergyIndex,
		greensFunction.getFundamentalMatsubaraEnergy()
	);

	#pragma omp parallel for
	for(unsigned int kx = 0; kx < numMeshPoints[0]; kx++){
		for(unsigned int ky = 0; ky < numMeshPoints[1]; ky++){
			for(
				unsigned int n = 0;
				n < numMatsubaraEnergiesSusceptibility;
				n++
			){
				int energyIndex
					= (lowerMatsubaraEnergyIndex + 2*(int)n)/2;

				if(
					energyIndex
						< -(int)numMatsubaraEnergiesGreensFunction
					|| energyIndex
						> (int)numMatsubaraEnergiesGreensFunction
				){
					susceptibility(
						{
							{kx, ky},
							intraBlockIndices[0],
							intraBlockIndices[1],
							intraBlockIndices[2],
							intraBlockIndices[3]
						},
						n
					) = 0;
				}
				else{
					if(energyIndex < 0)
						energyIndex += 2*(int)numMatsubaraEnergiesGreensFunction;

					susceptibility(
						{
							{kx, ky},
							intraBlockIndices[0],
							intraBlockIndices[1],
							intraBlockIndices[2],
							intraBlockIndices[3]
						},
						n
					) = -susceptibilityOut[
						2*numMatsubaraEnergiesGreensFunction*(
							numMeshPoints[1]*kx + ky
						) + energyIndex
					];
				}
			}
		}
	}

	return susceptibility;
}

}	//End of namespace Solver
}	//End of namesapce TBTK
