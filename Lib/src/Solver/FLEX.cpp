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
#include "TBTK/PropertyExtractor/Greens.h"
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

FLEX::FLEX(const RPA::MomentumSpaceContext &momentumSpaceContext) :
	momentumSpaceContext(momentumSpaceContext),
	reducedMomentumSpaceContext(
		momentumSpaceContext.getBrillouinZone(),
		momentumSpaceContext.getNumMeshPoints()
	)
{
	density = 0;
	targetDensity = -1;
	densityTolerance = 1e-5;
	chemicalPotentialStepSize = 1e-1;

	lowerFermionicMatsubaraEnergyIndex = -1;
	upperFermionicMatsubaraEnergyIndex = 1;
	lowerBosonicMatsubaraEnergyIndex = 0;
	upperBosonicMatsubaraEnergyIndex = 0;

	U = 0;
	J = 0;
	Up = 0;
	Jp = 0;

	state = State::NotYetStarted;
	maxIterations = 1;
	callback = nullptr;

	norm = Norm::Max;
	convergenceParameter = 0;
}

void FLEX::run(){
	//Calculate the non-interacting Green's function.
	Timer::tick("Green's function 0");
	calculateBareGreensFunction();
	greensFunction = greensFunction0;
	Timer::tock();

	state = State::GreensFunctionCalculated;
	if(callback != nullptr)
		callback(*this);

	//The main loop.
	unsigned int iteration = 0;
	while(iteration++ < maxIterations){
		//Calculate the bare susceptibility.
		calculateBareSusceptibility();
		state = State::BareSusceptibilityCalculated;
		if(callback != nullptr)
			callback(*this);

		//Calculate the RPA charge and spin susceptibilities.
		calculateRPASusceptibilities();
		state = State::RPASusceptibilitiesCalculated;
		if(callback != nullptr)
			callback(*this);

		//Calculate the interaction vertex.
		calculateInteractionVertex();
		state = State::InteractionVertexCalculated;
		if(callback != nullptr)
			callback(*this);

		//Calculate the self-energy.
		calculateSelfEnergy();
		state = State::SelfEnergyCalculated;
		if(callback != nullptr)
			callback(*this);

		//Calculate the Green's function.
		oldGreensFunction = greensFunction;
		calculateGreensFunction();
		state = State::GreensFunctionCalculated;
		if(callback != nullptr)
			callback(*this);

		calculateConvergenceParameter();
		if(convergenceParameter < tolerance)
			break;
	}
}

void FLEX::calculateBareGreensFunction(){
	const vector<unsigned int> &numMeshPoints
		= momentumSpaceContext.getNumMeshPoints();
	TBTKAssert(
		numMeshPoints.size() == 2,
		"Solver::FLEX::calculateBAreGreensFunction()()",
		"Only two-dimensional block indices supported yet, but the"
		<< " MomentumSpaceContext has a '" << numMeshPoints.size()
		<< "'-dimensional block structure.",
		""
	);

	BlockDiagonalizer blockDiagonalizer;
	blockDiagonalizer.setVerbose(false);
	blockDiagonalizer.setModel(getModel());
	blockDiagonalizer.run();

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
	greensFunction0
		= blockDiagonalizerPropertyExtractor.calculateGreensFunction(
			greensFunctionPatterns,
			Property::GreensFunction::Type::Matsubara
		);
}

void FLEX::calculateBareSusceptibility(){
	MatsubaraSusceptibility matsubaraSusceptibilitySolver(
		reducedMomentumSpaceContext,
		greensFunction
	);
	matsubaraSusceptibilitySolver.setVerbose(false);
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
}

void FLEX::calculateRPASusceptibilities(){
	RPASusceptibility rpaSusceptibilitySolver(
		reducedMomentumSpaceContext,
		bareSusceptibility
	);
	rpaSusceptibilitySolver.setVerbose(false);
	rpaSusceptibilitySolver.setModel(getModel());
	rpaSusceptibilitySolver.setU(U);
	rpaSusceptibilitySolver.setJ(J);
	rpaSusceptibilitySolver.setUp(Up);
	rpaSusceptibilitySolver.setJp(Jp);
	rpaSusceptibilitySolver.setNumOrbitals(
		momentumSpaceContext.getNumOrbitals()
	);

	PropertyExtractor::RPASusceptibility
		rpaSusceptibilityPropertyExtractor(
			rpaSusceptibilitySolver
		);

	rpaSusceptibilitySolver.setInteractionAmplitudes(
		generateRPAChargeSusceptibilityInteractionAmplitudes()
	);
	rpaChargeSusceptibility
		= rpaSusceptibilityPropertyExtractor.calculateRPASusceptibility({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});

	rpaSusceptibilitySolver.setInteractionAmplitudes(
		generateRPASpinSusceptibilityInteractionAmplitudes()
	);
	rpaSpinSusceptibility
		= rpaSusceptibilityPropertyExtractor.calculateRPASusceptibility({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});
}

void FLEX::calculateInteractionVertex(){
	ElectronFluctuationVertex
		electronFluctuationVertexSolver(
			momentumSpaceContext,
			rpaChargeSusceptibility,
			rpaSpinSusceptibility
		);
	electronFluctuationVertexSolver.setVerbose(false);
	electronFluctuationVertexSolver.setModel(getModel());
	electronFluctuationVertexSolver.setU(U);
	electronFluctuationVertexSolver.setJ(J);
	electronFluctuationVertexSolver.setUp(Up);
	electronFluctuationVertexSolver.setJp(Jp);

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
}

void FLEX::calculateSelfEnergy(){
	SelfEnergy2 selfEnergySolver(
		momentumSpaceContext,
		interactionVertex,
		greensFunction
	);
	selfEnergySolver.setVerbose(false);
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
	selfEnergy = selfEnergyPropertyExtractor.calculateSelfEnergy({
		{{IDX_ALL, IDX_ALL}, {IDX_ALL}, {IDX_ALL}}
	});
	convertSelfEnergyIndexStructure();
}

void FLEX::calculateGreensFunction(){
	if(targetDensity < 0){
		Greens greensSolver;
		greensSolver.setVerbose(false);
		greensSolver.setModel(getModel());
		greensSolver.setGreensFunction(greensFunction0);
		greensFunction = greensSolver.calculateInteractingGreensFunction(
			selfEnergy
		);
	}
	else{
		double chemicalPotential = getModel().getChemicalPotential();

		chemicalPotentialStepSize /= 2.;
		int previousChemicalPotentialStepDirection = 0;
		double chemicalPotentialStepMultiplier = 2.;
		double densityDifference = 2*densityTolerance;
		while(abs(densityDifference) > densityTolerance){
			Greens greensSolver;
			greensSolver.setVerbose(false);
			greensSolver.setModel(getModel());
			greensSolver.setGreensFunction(greensFunction0);
			greensFunction
				= greensSolver.calculateInteractingGreensFunction(
					selfEnergy
				);

			calculateDensity();
			densityDifference = density - targetDensity;
			if(abs(densityDifference) > densityTolerance){
				if(densityDifference > 0){
					switch(previousChemicalPotentialStepDirection){
					case 0:
						previousChemicalPotentialStepDirection = -1;
						break;
					case 1:
						previousChemicalPotentialStepDirection = -1;
						chemicalPotentialStepMultiplier = 0.5;
						break;
					case -1:
						previousChemicalPotentialStepDirection = -1;
						break;
					default:
						TBTKExit(
							"Solver::FLEX::calculateGreensFunction()",
							"Unknown step"
							<< " direction.",
							"This should never"
							<< " happen, contact"
							<< " the developer."
						);
					}

					chemicalPotentialStepSize
						*= chemicalPotentialStepMultiplier;

					chemicalPotential
						-= chemicalPotentialStepSize;
				}
				else{
					switch(previousChemicalPotentialStepDirection){
					case 0:
						previousChemicalPotentialStepDirection = 1;
						break;
					case 1:
						previousChemicalPotentialStepDirection = 1;
						break;
					case -1:
						previousChemicalPotentialStepDirection = 1;
						chemicalPotentialStepMultiplier = 0.5;
						break;
					default:
						TBTKExit(
							"Solver::FLEX::calculateGreensFunction()",
							"Unknown step"
							<< " direction.",
							"This should never"
							<< " happen, contact"
							<< " the developer."
						);
					}

					chemicalPotentialStepSize
						*= chemicalPotentialStepMultiplier;

					chemicalPotential
						+= chemicalPotentialStepSize;
				}
				getModel().setChemicalPotential(
					chemicalPotential
				);

				calculateBareGreensFunction();
			}

//			Streams::out << "Density:\t" << density << "\t" << chemicalPotential << "\n";
		}
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

void FLEX::calculateConvergenceParameter(){
	const vector<complex<double>> &oldData = oldGreensFunction.getData();
	const vector<complex<double>> &newData = greensFunction.getData();

	TBTKAssert(
		oldData.size() == newData.size(),
		"Solver::FLEX::calculateConvergenceParameter()",
		"Incompatible Green's function data sizes.",
		"This should never happen, contact the developer."
	);

	switch(norm){
	case Norm::Max:
	{
		double oldMax = 0;
		double differenceMax = 0;
		for(unsigned int n = 0; n < oldData.size(); n++){
			if(abs(oldData[n]) > oldMax)
				oldMax = abs(oldData[n]);
			if(abs(oldData[n] - newData[n]) > differenceMax)
				differenceMax = abs(oldData[n] - newData[n]);
		}

		convergenceParameter = differenceMax/oldMax;

		break;
	}
	case Norm::L2:
	{
		double oldL2 = 0;
		double differenceL2 = 0;
		for(unsigned int n = 0; n < oldData.size(); n++){
			oldL2 += pow(abs(oldData[n]), 2);
			differenceL2 += pow(abs(oldData[n] - newData[n]), 2);
		}

		convergenceParameter = differenceL2/oldL2;

		break;
	}
	default:
		TBTKExit(
			"Solver::FLEX::calculateConvergenceParameter()",
			"Unknown norm.",
			"This should never happen, contact the developer."
		);
	}
}

vector<InteractionAmplitude>
FLEX::generateRPAChargeSusceptibilityInteractionAmplitudes(){
	vector<InteractionAmplitude> interactionAmplitudes;
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	for(int a = 0; a < (int)numOrbitals; a++){
		interactionAmplitudes.push_back(
			InteractionAmplitude(
				U,
				{{a},	{a}},
				{{a},	{a}}
			)
		);

		for(int b = 0; b < (int)numOrbitals; b++){
			if(a == b)
				continue;

			interactionAmplitudes.push_back(
				InteractionAmplitude(
					2.*Up - J,
					{{a},	{b}},
					{{b},	{a}}
				)
			);
			interactionAmplitudes.push_back(
				InteractionAmplitude(
					-Up + 2.*J,
					{{a},	{b}},
					{{a},	{b}}
				)
			);
			interactionAmplitudes.push_back(
				InteractionAmplitude(
					Jp,
					{{a},	{a}},
					{{b},	{b}}
				)
			);
		}
	}

	return interactionAmplitudes;
}

vector<InteractionAmplitude>
FLEX::generateRPASpinSusceptibilityInteractionAmplitudes(){
	vector<InteractionAmplitude> interactionAmplitudes;
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	for(int a = 0; a < (int)numOrbitals; a++){
		interactionAmplitudes.push_back(
			InteractionAmplitude(
				-U,
				{{a},	{a}},
				{{a},	{a}}
			)
		);

		for(int b = 0; b < (int)numOrbitals; b++){
			if(a == b)
				continue;

			interactionAmplitudes.push_back(
				InteractionAmplitude(
					-J,
					{{a},	{b}},
					{{b},	{a}}
				)
			);
			interactionAmplitudes.push_back(
				InteractionAmplitude(
					-Up,
					{{a},	{b}},
					{{a},	{b}}
				)
			);
			interactionAmplitudes.push_back(
				InteractionAmplitude(
					-Jp,
					{{a},	{a}},
					{{b},	{b}}
				)
			);
		}
	}

	return interactionAmplitudes;
}

void FLEX::calculateDensity(){
	const vector<unsigned int> &numMeshPoints
		= momentumSpaceContext.getNumMeshPoints();

	Greens solver;
	solver.setModel(getModel());
	solver.setGreensFunction(greensFunction);

	PropertyExtractor::Greens propertyExtractor(solver);
	Property::Density densityProperty
		= propertyExtractor.calculateDensity({
			{IDX_SUM_ALL, IDX_SUM_ALL, IDX_SUM_ALL}
		});
	density = densityProperty({0, 0, 0});

	solver.setGreensFunction(greensFunction0);
	densityProperty = propertyExtractor.calculateDensity({
		{IDX_SUM_ALL, IDX_SUM_ALL, IDX_SUM_ALL}
	});
	density -= densityProperty({0, 0, 0});

	BlockDiagonalizer blockDiagonalizer;
	blockDiagonalizer.setVerbose(false);
	blockDiagonalizer.setModel(getModel());
	blockDiagonalizer.run();

	PropertyExtractor::BlockDiagonalizer
		blockDiagonalizerPropertyExtractor(blockDiagonalizer);
	densityProperty = blockDiagonalizerPropertyExtractor.calculateDensity({
		{IDX_SUM_ALL, IDX_SUM_ALL, IDX_SUM_ALL}
	});
	density += densityProperty({0, 0, 0});

	density *= 2./(numMeshPoints[0]*numMeshPoints[1]);
//	Streams::out << "Density:\t" << averageDensity << "\n";

//	return averageDensity;
}

}	//End of namespace Solver
}	//End of namesapce TBTK
