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

/** @file FLEX.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PropertyExtractor/BlockDiagonalizer.h"
#include "TBTK/PropertyExtractor/ElectronFluctuationVertex.h"
#include "TBTK/PropertyExtractor/MatsubaraSusceptibility.h"
#include "TBTK/PropertyExtractor/RPASusceptibility.h"
#include "TBTK/PropertyExtractor/SelfEnergy2.h"
#include "TBTK/Solver/BlockDiagonalizer.h"
#include "TBTK/Solver/ElectronFluctuationVertex.h"
#include "TBTK/Solver/Greens.h"
#include "TBTK/Solver/MatsubaraSusceptibility.h"
#include "TBTK/Solver/RPASusceptibility.h"
#include "TBTK/Solver/SelfEnergy2.h"
#include "TBTK/Solver/FLEX.h"

#include <complex>

using namespace std;

namespace TBTK{
namespace Solver{

FLEX::FLEX(const MomentumSpaceContext &momentumSpaceContext) :
	momentumSpaceContext(momentumSpaceContext)
{
	lowerFermionicMatsubaraEnergyIndex = -1;
	upperFermionicMatsubaraEnergyIndex = 1;
	lowerBosonicMatsubaraEnergyIndex = 0;
	upperBosonicMatsubaraEnergyIndex = 0;

	U = 0;
	J = 0;

	state = State::NotYetStarted;
	maxIterations = 1;
	callback = nullptr;
}

void FLEX::run(){
	const vector<unsigned int> &numMeshPoints
		= momentumSpaceContext.getNumMeshPoints();
	TBTKAssert(
		numMeshPoints.size() == 2,
		"Solver::FLEX::run()",
		"Only two-dimensional block indices supported yet, but the"
		<< " MomentumSpaceContext has a '" << numMeshPoints.size()
		<< "'-dimensional block structure.",
		""
	);

	BlockDiagonalizer blockDiagonalizer;
	blockDiagonalizer.setModel(getModel());
	blockDiagonalizer.run();

	//Calculate the non-interacting Green's function.
	Timer::tick("Green's function 0");
	vector<Index> greensFunctionPatterns;
	for(int kx = 0; kx < (int)numMeshPoints[0]; kx++){
		for(int ky = 0; ky < (int)numMeshPoints[1]; ky++){
			greensFunctionPatterns.push_back(
				{{kx, ky, IDX_ALL}, {kx, ky, IDX_ALL}}
			);
		}
	}

	PropertyExtractor::BlockDiagonalizer
		blockDiagonalizerPropertyExtractor(blockDiagonalizer);
	blockDiagonalizerPropertyExtractor.setEnergyWindow(
		lowerFermionicMatsubaraEnergyIndex,
		upperFermionicMatsubaraEnergyIndex,
		lowerBosonicMatsubaraEnergyIndex,
		upperBosonicMatsubaraEnergyIndex
	);
	Property::GreensFunction greensFunction0
		= blockDiagonalizerPropertyExtractor.calculateGreensFunction(
			greensFunctionPatterns,
			Property::GreensFunction::Type::Matsubara
		);
	greensFunction = greensFunction0;
	Timer::tock();

	state = State::GreensFunctionCalculated;
	if(callback != nullptr)
		callback(*this);

	//The main loop.
	unsigned int iteration = 0;
	while(iteration++ < maxIterations){
		Timer::tick("One iteration");
		//Calculate the bare susceptibility.
		Timer::tick("Bare susceptibility");
		MatsubaraSusceptibility matsubaraSusceptibilitySolver(
			momentumSpaceContext,
			greensFunction
		);
		matsubaraSusceptibilitySolver.setModel(getModel());

		PropertyExtractor::MatsubaraSusceptibility
			matsubaraSusceptibilityPropertyExtractor(
				matsubaraSusceptibilitySolver
			);
		matsubaraSusceptibilityPropertyExtractor.setEnergyWindow(
			lowerFermionicMatsubaraEnergyIndex,
			upperFermionicMatsubaraEnergyIndex,
			lowerBosonicMatsubaraEnergyIndex,
			upperBosonicMatsubaraEnergyIndex
		);
		bareSusceptibility
			= matsubaraSusceptibilityPropertyExtractor.calculateSusceptibility({
				{
					{IDX_ALL, IDX_ALL},
					{IDX_ALL},
					{IDX_ALL},
					{IDX_ALL},
					{IDX_ALL}
				}
			});
		Timer::tock();

		state = State::BareSusceptibilityCalculated;
		if(callback != nullptr)
			callback(*this);

		//Calculate the RPA charge and spin susceptibilities.
		Timer::tick("RPA susceptibilities");
		RPASusceptibility rpaSusceptibilitySolver(
			momentumSpaceContext,
			bareSusceptibility
		);
		rpaSusceptibilitySolver.setModel(getModel());
		rpaSusceptibilitySolver.setU(U);
		rpaSusceptibilitySolver.setJ(J);
		rpaSusceptibilitySolver.setUp(U - 2.*J);
		rpaSusceptibilitySolver.setJp(J);

		PropertyExtractor::RPASusceptibility
			rpaSusceptibilityPropertyExtractor(
				rpaSusceptibilitySolver
			);
		rpaChargeSusceptibility
			= rpaSusceptibilityPropertyExtractor.calculateChargeSusceptibility({
				{
					{IDX_ALL, IDX_ALL},
					{IDX_ALL},
					{IDX_ALL},
					{IDX_ALL},
					{IDX_ALL}
				}
			});
		rpaSpinSusceptibility
			= rpaSusceptibilityPropertyExtractor.calculateSpinSusceptibility({
				{
					{IDX_ALL, IDX_ALL},
					{IDX_ALL},
					{IDX_ALL},
					{IDX_ALL},
					{IDX_ALL}
				}
			});
		Timer::tock();

		state = State::RPASusceptibilitiesCalculated;
		if(callback != nullptr)
			callback(*this);

		//Calculate the interaction vertex.
		Timer::tick("Interaction vertex");
		ElectronFluctuationVertex
			electronFluctuationVertexSolver(
				momentumSpaceContext,
				rpaChargeSusceptibility,
				rpaSpinSusceptibility
			);
		electronFluctuationVertexSolver.setModel(getModel());
		electronFluctuationVertexSolver.setU(U);
		electronFluctuationVertexSolver.setJ(J);
		electronFluctuationVertexSolver.setUp(U - 2.*J);
		electronFluctuationVertexSolver.setJp(J);

		PropertyExtractor::ElectronFluctuationVertex
			electronFluctuationVertexPropertyExtractor(
				electronFluctuationVertexSolver
			);
		interactionVertex
			= electronFluctuationVertexPropertyExtractor.calculateInteractionVertex({
				{
					{IDX_ALL, IDX_ALL},
					{IDX_ALL},
					{IDX_ALL},
					{IDX_ALL},
					{IDX_ALL}
				}
			});
		Timer::tock();

		state = State::InteractionVertexCalculated;
		if(callback != nullptr)
			callback(*this);

		//Calculate the self-energy.
		Timer::tick("Self-energy");
		SelfEnergy2 selfEnergySolver(
			momentumSpaceContext,
			interactionVertex,
			greensFunction
		);
		selfEnergySolver.setModel(getModel());

		PropertyExtractor::SelfEnergy2 selfEnergyPropertyExtractor(
			selfEnergySolver
		);
		selfEnergyPropertyExtractor.setEnergyWindow(
			lowerFermionicMatsubaraEnergyIndex,
			upperFermionicMatsubaraEnergyIndex,
			lowerBosonicMatsubaraEnergyIndex,
			upperBosonicMatsubaraEnergyIndex
		);
		selfEnergy
			= selfEnergyPropertyExtractor.calculateSelfEnergy({
				{{IDX_ALL, IDX_ALL}, {IDX_ALL}, {IDX_ALL}}
			});
		convertSelfEnergyIndexStructure();
		Timer::tock();

		state = State::SelfEnergyCalculated;
		if(callback != nullptr)
			callback(*this);

		//Calculate the self energy.
		Timer::tick("Green's function");
		Greens greensSolver;
		greensSolver.setModel(getModel());
		greensSolver.setGreensFunction(greensFunction0);
		greensFunction
			= greensSolver.calculateInteractingGreensFunction(
				selfEnergy
			);
		Timer::tock();

		state = State::GreensFunctionCalculated;
		if(callback != nullptr)
			callback(*this);

		Timer::tock();
	}
}

void FLEX::convertSelfEnergyIndexStructure(){
	const vector<unsigned int> &numMeshPoints
		= momentumSpaceContext.getNumMeshPoints();
	TBTKAssert(
		numMeshPoints.size() == 2,
		"Solver::FLEX::convertSelfEnergyBlockStructure()",
		"Only two-dimensional block indices supported yet, but the"
		<< " MomentumSpaceContext has a '" << numMeshPoints.size()
		<< "'-dimensional block structure.",
		""
	);
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	IndexTree memoryLayout;
	for(unsigned int kx = 0; kx < numMeshPoints[0]; kx++){
		for(unsigned int ky = 0; ky < numMeshPoints[1]; ky++){
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
					memoryLayout.add({
						{(int)kx, (int)ky, (int)orbital0},
						{(int)kx, (int)ky, (int)orbital1}
					});
				}
			}
		}
	}
	memoryLayout.generateLinearMap();
	Property::SelfEnergy newSelfEnergy(
		memoryLayout,
		selfEnergy.getLowerMatsubaraEnergyIndex(),
		selfEnergy.getUpperMatsubaraEnergyIndex(),
		selfEnergy.getFundamentalMatsubaraEnergy()
	);
	for(unsigned int kx = 0; kx < numMeshPoints[0]; kx++){
		for(unsigned int ky = 0; ky < numMeshPoints[1]; ky++){
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
					for(
						unsigned int n = 0;
						n < selfEnergy.getNumMatsubaraEnergies();
						n++
					){
						newSelfEnergy(
							{
								{
									(int)kx,
									(int)ky,
									(int)orbital0
								},
								{
									(int)kx,
									(int)ky,
									(int)orbital1
								}
							},
							n
						) = selfEnergy(
							{
								{
									(int)kx,
									(int)ky
								},
								{(int)orbital0},
								{(int)orbital1}
							},
							n
						);
					}
				}
			}
		}
	}

	selfEnergy = newSelfEnergy;
}

}	//End of namespace Solver
}	//End of namesapce TBTK
