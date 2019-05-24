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

FLEX::FLEX(
	const MomentumSpaceContext &momentumSpaceContext
) :
	momentumSpaceContext(momentumSpaceContext)
{
	selfEnergyMixingParameter = 0;
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

	numOrbitals = 0;

	state = State::NotYetStarted;
	maxIterations = 1;
	callback = nullptr;

	norm = Norm::Max;
	tolerance = 0;
	convergenceParameter = 0;
	numSlices = 1;
}

void FLEX::run(){
	//Calculate the non-interacting Green's function.
	Timer::tick("Green's function 0");
	calculateBareGreensFunction();
	greensFunction = greensFunction0;
	if(selfEnergy.getData().size() != 0)
		calculateGreensFunction();
	Timer::tock();

	state = State::GreensFunctionCalculated;
	if(callback != nullptr)
		callback(*this);

	//The main loop.
	unsigned int iteration = 0;
	while(iteration++ < maxIterations){
		for(unsigned int n = 0; n < numSlices; n++){
			Timer::tick("One slice");

			Timer::tick("Bare susceptibility");
			//Calculate the bare susceptibility.
			calculateBareSusceptibility(n);
			state = State::BareSusceptibilityCalculated;
			if(callback != nullptr)
				callback(*this);
			Timer::tock();

			Timer::tick("RPA susceptibility");
			//Calculate the RPA charge and spin susceptibilities.
			calculateRPASusceptibilities();
			state = State::RPASusceptibilitiesCalculated;
			if(callback != nullptr)
				callback(*this);
			Timer::tock();

			Timer::tick("Interaction vertex");
			//Calculate the interaction vertex.
			calculateInteractionVertex();
			state = State::InteractionVertexCalculated;
			if(callback != nullptr)
				callback(*this);
			Timer::tock();

			Timer::tick("Self energy");
			//Calculate the self-energy.
			calculateSelfEnergy(n);
			state = State::SelfEnergyCalculated;
			if(callback != nullptr)
				callback(*this);
			Timer::tock();

			Timer::tock();
		}

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

void FLEX::calculateBareSusceptibility(unsigned int slice){
	MatsubaraSusceptibility matsubaraSusceptibilitySolver(
		momentumSpaceContext,
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
		getLowerBosonicMatsubaraEnergyIndex(slice),
		getUpperBosonicMatsubaraEnergyIndex(slice)
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
		momentumSpaceContext,
		bareSusceptibility
	);
	rpaSusceptibilitySolver.setVerbose(false);
	rpaSusceptibilitySolver.setModel(getModel());

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
	//////////////////////////
	// Charge contributions //
	//////////////////////////
	ElectronFluctuationVertex electronFluctuationVertexChargeSolver(
		momentumSpaceContext,
		rpaChargeSusceptibility
	);
	electronFluctuationVertexChargeSolver.setVerbose(false);
	electronFluctuationVertexChargeSolver.setModel(getModel());
	PropertyExtractor::ElectronFluctuationVertex
		electronFluctuationVertexChargePropertyExtractor(
			electronFluctuationVertexChargeSolver
		);

	//U_1*\chi_c*U_1
	electronFluctuationVertexChargeSolver.setLeftInteraction(generateU1());
	electronFluctuationVertexChargeSolver.setRightInteraction(generateU1());
	interactionVertex
		= (1/2.)*electronFluctuationVertexChargePropertyExtractor.calculateInteractionVertex({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});

	//U_2*\chi_c*U_2
	electronFluctuationVertexChargeSolver.setLeftInteraction(generateU2());
	electronFluctuationVertexChargeSolver.setRightInteraction(generateU2());
	interactionVertex
		+= (1/2.)*electronFluctuationVertexChargePropertyExtractor.calculateInteractionVertex({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});

	//U_1*\chi_c*U_2
	electronFluctuationVertexChargeSolver.setLeftInteraction(generateU1());
	electronFluctuationVertexChargeSolver.setRightInteraction(generateU2());
	interactionVertex
		+= (1/2.)*electronFluctuationVertexChargePropertyExtractor.calculateInteractionVertex({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});

	//U_2*\chi_c*U_1
	electronFluctuationVertexChargeSolver.setLeftInteraction(generateU2());
	electronFluctuationVertexChargeSolver.setRightInteraction(generateU1());
	interactionVertex
		+= (1/2.)*electronFluctuationVertexChargePropertyExtractor.calculateInteractionVertex({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});

	////////////////////////
	// Spin contributions //
	////////////////////////
	ElectronFluctuationVertex electronFluctuationVertexSpinSolver(
		momentumSpaceContext,
		rpaSpinSusceptibility
	);
	electronFluctuationVertexSpinSolver.setVerbose(false);
	electronFluctuationVertexSpinSolver.setModel(getModel());
	PropertyExtractor::ElectronFluctuationVertex
		electronFluctuationVertexSpinPropertyExtractor(
			electronFluctuationVertexSpinSolver
		);

	//U_1*\chi_s*U_1
	electronFluctuationVertexSpinSolver.setLeftInteraction(generateU1());
	electronFluctuationVertexSpinSolver.setRightInteraction(generateU1());
	interactionVertex
		+= (1/2.)*electronFluctuationVertexSpinPropertyExtractor.calculateInteractionVertex({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});

	//U_2*\chi_s*U_2
	electronFluctuationVertexSpinSolver.setLeftInteraction(generateU2());
	electronFluctuationVertexSpinSolver.setRightInteraction(generateU2());
	interactionVertex
		+= (1/2.)*electronFluctuationVertexSpinPropertyExtractor.calculateInteractionVertex({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});

	//U_1*\chi_s*U_2
	electronFluctuationVertexSpinSolver.setLeftInteraction(generateU1());
	electronFluctuationVertexSpinSolver.setRightInteraction(generateU2());
	interactionVertex
		+= (-1/2.)*electronFluctuationVertexSpinPropertyExtractor.calculateInteractionVertex({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});

	//U_2*\chi_s*U_1
	electronFluctuationVertexSpinSolver.setLeftInteraction(generateU2());
	electronFluctuationVertexSpinSolver.setRightInteraction(generateU1());
	interactionVertex
		+= (-1/2.)*electronFluctuationVertexSpinPropertyExtractor.calculateInteractionVertex({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});

	//U_3*\chi_s*U_3
	electronFluctuationVertexSpinSolver.setLeftInteraction(generateU3());
	electronFluctuationVertexSpinSolver.setRightInteraction(generateU3());
	interactionVertex
		+= electronFluctuationVertexSpinPropertyExtractor.calculateInteractionVertex({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});

	//////////////////////////
	// Double counitng term //
	//////////////////////////
	ElectronFluctuationVertex electronFluctuationVertexDoubleCountingSolver(
		momentumSpaceContext,
		bareSusceptibility
	);
	electronFluctuationVertexDoubleCountingSolver.setVerbose(false);
	electronFluctuationVertexDoubleCountingSolver.setModel(getModel());
	PropertyExtractor::ElectronFluctuationVertex
		electronFluctuationVertexDoubleCountingPropertyExtractor(
			electronFluctuationVertexDoubleCountingSolver
		);

	//U_4*\chi_b*U_4
	electronFluctuationVertexDoubleCountingSolver.setLeftInteraction(generateU4());
	electronFluctuationVertexDoubleCountingSolver.setRightInteraction(generateU4());
	interactionVertex
		-= electronFluctuationVertexDoubleCountingPropertyExtractor.calculateInteractionVertex({
			{
				{IDX_ALL, IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL},
				{IDX_ALL}
			}
		});

	///////////////////////
	// Hartree-Fock term //
	///////////////////////
	interactionVertex += generateHartreeFockTerm();
}

void FLEX::calculateSelfEnergy(unsigned int slice){
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
	if(slice == 0){
		previousSelfEnergy = selfEnergy;
		selfEnergy = selfEnergyPropertyExtractor.calculateSelfEnergy({
			{{IDX_ALL, IDX_ALL}, {IDX_ALL}, {IDX_ALL}}
		});
	}
	else{
		selfEnergy += selfEnergyPropertyExtractor.calculateSelfEnergy({
			{{IDX_ALL, IDX_ALL}, {IDX_ALL}, {IDX_ALL}}
		});
	}

	if(slice == numSlices - 1){
		convertSelfEnergyIndexStructure();

		if(
			previousSelfEnergy.getData().size() != 0
			&& selfEnergyMixingParameter != 0
		){
			selfEnergy = (1 - selfEnergyMixingParameter)*selfEnergy
				+ selfEnergyMixingParameter*previousSelfEnergy;
		}
	}
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
	TBTKAssert(
		numOrbitals != 0,
		"Solver::FLEX::convertSelfEnergyIndexStructure()",
		"'numOrbitals' must be non-zero.",
		"Use Solver::FLEX::setNumOrbitals() to set the number of"
		<< " orbitals."
	);

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
	TBTKAssert(
		numOrbitals != 0,
		"Solver::FLEX::generateRPAChargeSusceptibilityInteractionAmplitudes()",
		"'numOrbitals' must be non-zero.",
		"Use Solver::FLEX::setNumOrbitals() to set the number of"
		<< " orbitals."
	);

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
	TBTKAssert(
		numOrbitals != 0,
		"Solver::FLEX::generateRPASpinSusceptibilityInteractionAmplitudes()",
		"'numOrbitals' must be non-zero.",
		"Use Solver::FLEX::setNumOrbitals() to set the number of"
		<< " orbitals."
	);

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

vector<InteractionAmplitude> FLEX::generateU1(){
	vector<InteractionAmplitude> interactionAmplitudes;
	TBTKAssert(
		numOrbitals != 0,
		"Solver::FLEX::generateU1()",
		"'numOrbitals' must be non-zero.",
		"Use Solver::FLEX::setNumOrbitals() to set the number of"
		<< " orbitals."
	);

	for(int a = 0; a < (int)numOrbitals; a++){
		for(int b = 0; b < (int)numOrbitals; b++){
			if(a == b)
				continue;

			interactionAmplitudes.push_back(
				InteractionAmplitude(
					Up - J,
					{{a}, {b}},
					{{b}, {a}}
				)
			);
			interactionAmplitudes.push_back(
				InteractionAmplitude(
					J - Up,
					{{a}, {b}},
					{{a}, {b}}
				)
			);
		}
	}

	return interactionAmplitudes;
}

vector<InteractionAmplitude> FLEX::generateU2(){
	vector<InteractionAmplitude> interactionAmplitudes;
	TBTKAssert(
		numOrbitals != 0,
		"Solver::FLEX::generateU2()",
		"'numOrbitals' must be non-zero.",
		"Use Solver::FLEX::setNumOrbitals() to set the number of"
		<< " orbitals."
	);

	for(int a = 0; a < (int)numOrbitals; a++){
		interactionAmplitudes.push_back(
			InteractionAmplitude(
				U,
				{{a}, {a}},
				{{a}, {a}}
			)
		);

		for(int b = 0; b < (int)numOrbitals; b++){
			if(a == b)
				continue;

			interactionAmplitudes.push_back(
				InteractionAmplitude(
					Up,
					{{a}, {b}},
					{{b}, {a}}
				)
			);
			interactionAmplitudes.push_back(
				InteractionAmplitude(
					J,
					{{a}, {b}},
					{{a}, {b}}
				)
			);
			interactionAmplitudes.push_back(
				InteractionAmplitude(
					Jp,
					{{a}, {a}},
					{{b}, {b}}
				)
			);
		}
	}

	return interactionAmplitudes;
}

vector<InteractionAmplitude> FLEX::generateU3(){
	vector<InteractionAmplitude> interactionAmplitudes;
	TBTKAssert(
		numOrbitals != 0,
		"Solver::FLEX::generateU3()",
		"'numOrbitals' must be non-zero.",
		"Use Solver::FLEX::setNumOrbitals() to set the number of"
		<< " orbitals."
	);

	for(int a = 0; a < (int)numOrbitals; a++){
		interactionAmplitudes.push_back(
			InteractionAmplitude(
				-U,
				{{a}, {a}},
				{{a}, {a}}
			)
		);

		for(int b = 0; b < (int)numOrbitals; b++){
			if(a == b)
				continue;

			interactionAmplitudes.push_back(
				InteractionAmplitude(
					-J,
					{{a}, {b}},
					{{b}, {a}}
				)
			);
			interactionAmplitudes.push_back(
				InteractionAmplitude(
					-Up,
					{{a}, {b}},
					{{a}, {b}}
				)
			);
			interactionAmplitudes.push_back(
				InteractionAmplitude(
					-Jp,
					{{a}, {a}},
					{{b}, {b}}
				)
			);
		}
	}

	return interactionAmplitudes;
}

vector<InteractionAmplitude> FLEX::generateU4(){
	vector<InteractionAmplitude> interactionAmplitudes;
	TBTKAssert(
		numOrbitals != 0,
		"Solver::FLEX::generateU4()",
		"'numOrbitals' must be non-zero.",
		"Use Solver::FLEX::setNumOrbitals() to set the number of"
		<< " orbitals."
	);

	for(int a = 0; a < (int)numOrbitals; a++){
		interactionAmplitudes.push_back(
			InteractionAmplitude(
				U,
				{{a}, {a}},
				{{a}, {a}}
			)
		);

		for(int b = 0; b < (int)numOrbitals; b++){
			if(a == b)
				continue;

			interactionAmplitudes.push_back(
				InteractionAmplitude(
					Up,
					{{a}, {b}},
					{{b}, {a}}
				)
			);
			interactionAmplitudes.push_back(
				InteractionAmplitude(
					J,
					{{a}, {b}},
					{{a}, {b}}
				)
			);
			interactionAmplitudes.push_back(
				InteractionAmplitude(
					Jp,
					{{a}, {a}},
					{{b}, {b}}
				)
			);
		}
	}

	return interactionAmplitudes;
}

Property::InteractionVertex FLEX::generateHartreeFockTerm(){
	Property::InteractionVertex hartreeFockTerm
		= interactionVertex;

	vector<complex<double>> &data = hartreeFockTerm.getDataRW();
	for(unsigned int n = 0; n < data.size(); n++)
		data[n] = 0;

	const vector<unsigned int> &numMeshPoints
		= momentumSpaceContext.getNumMeshPoints();

	for(int kx = 0; kx < (int)numMeshPoints[0]; kx++){
		for(int ky = 0; ky < (int)numMeshPoints[1]; ky++){
			for(
				unsigned int n = 0;
				n < hartreeFockTerm.getNumEnergies();
				n++
			){
				for(int a = 0; a < (int)numOrbitals; a++){
					hartreeFockTerm(
						{
							{kx, ky},
							{a},
							{a},
							{a},
							{a}
						},
						n
					) += U;
					for(int b = 0; b < (int)numOrbitals; b++){
						if(a == b)
							continue;

						hartreeFockTerm(
							{
								{kx, ky},
								{a},
								{a},
								{b},
								{b}
							},
							n
						) += Up - 2*(Up - J);

						hartreeFockTerm(
							{
								{kx, ky},
								{a},
								{b},
								{b},
								{a}
							},
							n
						) += J - 2*(J - Up);

						hartreeFockTerm(
							{
								{kx, ky},
								{a},
								{b},
								{a},
								{b}
							},
							n
						) += Jp;
					}
				}
			}
		}
	}

	return hartreeFockTerm;
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
}

int FLEX::getLowerBosonicMatsubaraEnergyIndex(unsigned int slice){
	unsigned int numIndices = (
		upperBosonicMatsubaraEnergyIndex
		- lowerBosonicMatsubaraEnergyIndex
	)/2
	+ 1;
	unsigned int numIndicesPerSlice = numIndices/numSlices;

	return lowerBosonicMatsubaraEnergyIndex
		+ (int)(2*slice*numIndicesPerSlice);
}

int FLEX::getUpperBosonicMatsubaraEnergyIndex(unsigned int slice){
	if(slice == numSlices - 1)
		return upperBosonicMatsubaraEnergyIndex;

	unsigned int numIndices = (
		upperBosonicMatsubaraEnergyIndex
		- lowerBosonicMatsubaraEnergyIndex
	)/2
	+ 1;
	unsigned int numIndicesPerSlice = numIndices/numSlices;

	return lowerBosonicMatsubaraEnergyIndex
		+ (int)(2*(slice + 1)*numIndicesPerSlice)
		- 2;
}

}	//End of namespace Solver
}	//End of namesapce TBTK
