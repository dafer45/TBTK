/* Copyright 2017 Kristofer Björnson
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

/** @file SelfEnergyCalculator.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Functions.h"
#include "TBTK/InteractionAmplitude.h"
#include "TBTK/RPA/ElectronFluctuationVertexCalculator.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

//const complex<double> i(0, 1);

namespace TBTK{

ElectronFluctuationVertexCalculator::ElectronFluctuationVertexCalculator(
	const RPA::MomentumSpaceContext &momentumSpaceContext
){
	U = 0.;
	Up = 0.;
	J = 0.;
	Jp = 0.;

	rpaSusceptibilityCalculator = new RPASusceptibilityCalculator(
		momentumSpaceContext,
		SusceptibilityCalculator::Algorithm::Lindhard
	);

	interactionAmplitudesAreGenerated = false;

	isInitialized = true;
}

ElectronFluctuationVertexCalculator::ElectronFluctuationVertexCalculator(
	RPASusceptibilityCalculator &rpaSusceptibilityCalculator
){
	U = 0.;
	Up = 0.;
	J = 0.;
	Jp = 0.;

	this->rpaSusceptibilityCalculator = rpaSusceptibilityCalculator.createSlave();

	interactionAmplitudesAreGenerated = false;

	isInitialized = true;
}

ElectronFluctuationVertexCalculator::~ElectronFluctuationVertexCalculator(){
	if(rpaSusceptibilityCalculator != nullptr)
		delete rpaSusceptibilityCalculator;
}

ElectronFluctuationVertexCalculator* ElectronFluctuationVertexCalculator::createSlave(){
	return new ElectronFluctuationVertexCalculator(
		*rpaSusceptibilityCalculator
	);
}

void ElectronFluctuationVertexCalculator::generateInteractionAmplitudes(){
	if(interactionAmplitudesAreGenerated)
		return;

	u1.clear();
	u2.clear();
	u3.clear();

	const RPA::MomentumSpaceContext &momentumSpaceContext = rpaSusceptibilityCalculator->getMomentumSpaceContext();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	for(int a = 0; a < (int)numOrbitals; a++){
		u2.push_back(
			InteractionAmplitude(
				U,
				{{a},	{a}},
				{{a},	{a}}
			)
		);
		u3.push_back(
			InteractionAmplitude(
				-U,
				{{a},	{a}},
				{{a},	{a}}
			)
		);

		for(int b = 0; b < (int)numOrbitals; b++){
			if(a == b)
				continue;

			u1.push_back(
				InteractionAmplitude(
					Up - J,
					{{a},	{b}},
					{{b},	{a}}
				)
			);
			u2.push_back(
				InteractionAmplitude(
					Up,
					{{a},	{b}},
					{{b},	{a}}
				)
			);
			u3.push_back(
				InteractionAmplitude(
					-J,
					{{a},	{b}},
					{{b},	{a}}
				)
			);

			u1.push_back(
				InteractionAmplitude(
					J - Up,
					{{a},	{b}},
					{{a},	{b}}
				)
			);
			u2.push_back(
				InteractionAmplitude(
					J,
					{{a},	{b}},
					{{a},	{b}}
				)
			);
			u3.push_back(
				InteractionAmplitude(
					-Up,
					{{a},	{b}},
					{{a},	{b}}
				)
			);

			u2.push_back(
				InteractionAmplitude(
					Jp,
					{{a},	{a}},
					{{b},	{b}}
				)
			);
			u3.push_back(
				InteractionAmplitude(
					-Jp,
					{{a},	{a}},
					{{b},	{b}}
				)
			);
		}
	}

	interactionAmplitudesAreGenerated = true;
}

vector<complex<double>> ElectronFluctuationVertexCalculator::calculateSelfEnergyVertex(
	const vector<double> &k,
	const vector<int> &orbitalIndices
){
	TBTKAssert(
		isInitialized,
		"ElectronFLuctuationVertexCalculator::calculateSelfEnergyVertex()",
		"ElectronFluctuationVertexCalculator not yet initialized.",
		"Use ElectronFluctuationVertexCalculator::init() to initialize"
		<< " the SelfEnergyCalculator."
	);
	TBTKAssert(
		orbitalIndices.size() == 4,
		"ElectronFluctuationVertexCalculator::calculateSelfEnergyVertex()",
		"Two orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	const RPA::MomentumSpaceContext &momentumSpaceContext = rpaSusceptibilityCalculator->getMomentumSpaceContext();
	const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();

	//Calculate offset
	Index kIndex = brillouinZone.getMinorCellIndex(
		k,
		numMeshPoints
	);

	Index resultIndex = Index(
		kIndex,
		{
			orbitalIndices.at(0),
			orbitalIndices.at(1),
			orbitalIndices.at(2),
			orbitalIndices.at(3)
		}
	);

	SerializableVector<complex<double>> result;
	if(vertexTree.get(result, resultIndex))
		return result;

	DualIndex kDual(kIndex, k);

	vector<complex<double>> selfEnergyVertex;
	selfEnergyVertex.reserve(energies.size());
	for(
		unsigned int n = 0;
		n < energies.size();
		n++
	){
		selfEnergyVertex.push_back(0.);
	}

	//U_1*\chi_1*U_1
	for(unsigned int in = 0; in < u1.size(); in++){
		const InteractionAmplitude &incommingAmplitude = u1.at(in);
		complex<double> amplitude_i = incommingAmplitude.getAmplitude();
		int c0_i = incommingAmplitude.getCreationOperatorIndex(0).at(0);
		int c1_i = incommingAmplitude.getCreationOperatorIndex(1).at(0);
		int a0_i = incommingAmplitude.getAnnihilationOperatorIndex(0).at(0);
		int a1_i = incommingAmplitude.getAnnihilationOperatorIndex(1).at(0);

		if(
			a1_i != orbitalIndices.at(3)
			|| c0_i != orbitalIndices.at(2)
			|| abs(amplitude_i) < 1e-10
		){
			continue;
		}

		for(unsigned int out = 0; out < u1.size(); out++){
			const InteractionAmplitude &outgoingAmplitude = u1.at(out);
			complex<double> amplitude_o = outgoingAmplitude.getAmplitude();
			int c0_o = outgoingAmplitude.getCreationOperatorIndex(0).at(0);
			int c1_o = outgoingAmplitude.getCreationOperatorIndex(1).at(0);
			int a0_o = outgoingAmplitude.getAnnihilationOperatorIndex(0).at(0);
			int a1_o = outgoingAmplitude.getAnnihilationOperatorIndex(1).at(0);

			if(
				a0_o != orbitalIndices.at(0)
				|| c1_o != orbitalIndices.at(1)
				|| abs(amplitude_o) < 1e-10
			){
				continue;
			}

			vector<complex<double>> chargeSusceptibility = rpaSusceptibilityCalculator->calculateChargeRPASusceptibility(
				kDual,
				{c0_o, a1_o, c1_i, a0_i}
			);
			vector<complex<double>> spinSusceptibility = rpaSusceptibilityCalculator->calculateSpinRPASusceptibility(
				kDual,
				{c0_o, a1_o, c1_i, a0_i}
			);
			for(
				unsigned int n = 0;
				n < energies.size();
				n++
			){
				selfEnergyVertex.at(n) += amplitude_i*amplitude_o*(
					chargeSusceptibility.at(n)
					+ spinSusceptibility.at(n)
				)/2.;
			}
		}
	}

	//U_2*\chi_1*U_2
	for(unsigned int in = 0; in < u2.size(); in++){
		const InteractionAmplitude &incommingAmplitude = u2.at(in);
		complex<double> amplitude_i = incommingAmplitude.getAmplitude();
		int c0_i = incommingAmplitude.getCreationOperatorIndex(0).at(0);
		int c1_i = incommingAmplitude.getCreationOperatorIndex(1).at(0);
		int a0_i = incommingAmplitude.getAnnihilationOperatorIndex(0).at(0);
		int a1_i = incommingAmplitude.getAnnihilationOperatorIndex(1).at(0);

		if(
			a1_i != orbitalIndices.at(3)
			|| c0_i != orbitalIndices.at(2)
			|| abs(amplitude_i) < 1e-10
		){
			continue;
		}

		for(unsigned int out = 0; out < u2.size(); out++){
			const InteractionAmplitude &outgoingAmplitude = u2.at(out);
			complex<double> amplitude_o = outgoingAmplitude.getAmplitude();
			int c0_o = outgoingAmplitude.getCreationOperatorIndex(0).at(0);
			int c1_o = outgoingAmplitude.getCreationOperatorIndex(1).at(0);
			int a0_o = outgoingAmplitude.getAnnihilationOperatorIndex(0).at(0);
			int a1_o = outgoingAmplitude.getAnnihilationOperatorIndex(1).at(0);

			if(
				a0_o != orbitalIndices.at(0)
				|| c1_o != orbitalIndices.at(1)
				|| abs(amplitude_o) < 1e-10
			){
				continue;
			}

			vector<complex<double>> chargeSusceptibility = rpaSusceptibilityCalculator->calculateChargeRPASusceptibility(
				kDual,
				{c0_o, a1_o, c1_i, a0_i}
			);
			vector<complex<double>> spinSusceptibility = rpaSusceptibilityCalculator->calculateSpinRPASusceptibility(
				kDual,
				{c0_o, a1_o, c1_i, a0_i}
			);
			for(
				unsigned int n = 0;
				n < energies.size();
				n++
			){
				selfEnergyVertex.at(n) += amplitude_i*amplitude_o*(
					chargeSusceptibility.at(n)
					+ spinSusceptibility.at(n)
				)/2.;
			}
		}
	}

	//U_1*\chi_2*U_2
	for(unsigned int in = 0; in < u1.size(); in++){
		const InteractionAmplitude &incommingAmplitude = u1.at(in);
		complex<double> amplitude_i = incommingAmplitude.getAmplitude();
		int c0_i = incommingAmplitude.getCreationOperatorIndex(0).at(0);
		int c1_i = incommingAmplitude.getCreationOperatorIndex(1).at(0);
		int a0_i = incommingAmplitude.getAnnihilationOperatorIndex(0).at(0);
		int a1_i = incommingAmplitude.getAnnihilationOperatorIndex(1).at(0);

		if(
			a1_i != orbitalIndices.at(3)
			|| c0_i != orbitalIndices.at(2)
			|| abs(amplitude_i) < 1e-10
		){
			continue;
		}

		for(unsigned int out = 0; out < u2.size(); out++){
			const InteractionAmplitude &outgoingAmplitude = u2.at(out);
			complex<double> amplitude_o = outgoingAmplitude.getAmplitude();
			int c0_o = outgoingAmplitude.getCreationOperatorIndex(0).at(0);
			int c1_o = outgoingAmplitude.getCreationOperatorIndex(1).at(0);
			int a0_o = outgoingAmplitude.getAnnihilationOperatorIndex(0).at(0);
			int a1_o = outgoingAmplitude.getAnnihilationOperatorIndex(1).at(0);

			if(
				a0_o != orbitalIndices.at(0)
				|| c1_o != orbitalIndices.at(1)
				|| abs(amplitude_o) < 1e-10
			){
				continue;
			}

			vector<complex<double>> chargeSusceptibility = rpaSusceptibilityCalculator->calculateChargeRPASusceptibility(
				kDual,
				{c0_o, a1_o, c1_i, a0_i}
			);
			vector<complex<double>> spinSusceptibility = rpaSusceptibilityCalculator->calculateSpinRPASusceptibility(
				kDual,
				{c0_o, a1_o, c1_i, a0_i}
			);
			for(
				unsigned int n = 0;
				n < energies.size();
				n++
			){
				selfEnergyVertex.at(n) += amplitude_i*amplitude_o*(
					chargeSusceptibility.at(n)
					- spinSusceptibility.at(n)
				)/2.;
			}
		}
	}

	//U_2*\chi_2*U_1
	for(unsigned int in = 0; in < u2.size(); in++){
		const InteractionAmplitude &incommingAmplitude = u2.at(in);
		complex<double> amplitude_i = incommingAmplitude.getAmplitude();
		int c0_i = incommingAmplitude.getCreationOperatorIndex(0).at(0);
		int c1_i = incommingAmplitude.getCreationOperatorIndex(1).at(0);
		int a0_i = incommingAmplitude.getAnnihilationOperatorIndex(0).at(0);
		int a1_i = incommingAmplitude.getAnnihilationOperatorIndex(1).at(0);

		if(
			a1_i != orbitalIndices.at(3)
			|| c0_i != orbitalIndices.at(2)
			|| abs(amplitude_i) < 1e-10
		){
			continue;
		}

		for(unsigned int out = 0; out < u1.size(); out++){
			const InteractionAmplitude &outgoingAmplitude = u1.at(out);
			complex<double> amplitude_o = outgoingAmplitude.getAmplitude();
			int c0_o = outgoingAmplitude.getCreationOperatorIndex(0).at(0);
			int c1_o = outgoingAmplitude.getCreationOperatorIndex(1).at(0);
			int a0_o = outgoingAmplitude.getAnnihilationOperatorIndex(0).at(0);
			int a1_o = outgoingAmplitude.getAnnihilationOperatorIndex(1).at(0);

			if(
				a0_o != orbitalIndices.at(0)
				|| c1_o != orbitalIndices.at(1)
				|| abs(amplitude_o) < 1e-10
			){
				continue;
			}

			vector<complex<double>> chargeSusceptibility = rpaSusceptibilityCalculator->calculateChargeRPASusceptibility(
				kDual,
				{c0_o, a1_o, c1_i, a0_i}
			);
			vector<complex<double>> spinSusceptibility = rpaSusceptibilityCalculator->calculateSpinRPASusceptibility(
				kDual,
				{c0_o, a1_o, c1_i, a0_i}
			);
			for(
				unsigned int n = 0;
				n < energies.size();
				n++
			){
				selfEnergyVertex.at(n) += amplitude_i*amplitude_o*(
					chargeSusceptibility.at(n)
					- spinSusceptibility.at(n)
				)/2.;
			}
		}
	}

	//U_3*\chi_3*U_3
	for(unsigned int in = 0; in < u3.size(); in++){
		const InteractionAmplitude &incommingAmplitude = u3.at(in);
		complex<double> amplitude_i = incommingAmplitude.getAmplitude();
		int c0_i = incommingAmplitude.getCreationOperatorIndex(0).at(0);
		int c1_i = incommingAmplitude.getCreationOperatorIndex(1).at(0);
		int a0_i = incommingAmplitude.getAnnihilationOperatorIndex(0).at(0);
		int a1_i = incommingAmplitude.getAnnihilationOperatorIndex(1).at(0);

		if(
			a1_i != orbitalIndices.at(3)
			|| c0_i != orbitalIndices.at(2)
			|| abs(amplitude_i) < 1e-10
		){
			continue;
		}

		for(unsigned int out = 0; out < u3.size(); out++){
			const InteractionAmplitude &outgoingAmplitude = u3.at(out);
			complex<double> amplitude_o = outgoingAmplitude.getAmplitude();
			int c0_o = outgoingAmplitude.getCreationOperatorIndex(0).at(0);
			int c1_o = outgoingAmplitude.getCreationOperatorIndex(1).at(0);
			int a0_o = outgoingAmplitude.getAnnihilationOperatorIndex(0).at(0);
			int a1_o = outgoingAmplitude.getAnnihilationOperatorIndex(1).at(0);

			if(
				a0_o != orbitalIndices.at(0)
				|| c1_o != orbitalIndices.at(1)
				|| abs(amplitude_o) < 1e-10
			){
				continue;
			}

			vector<complex<double>> spinSusceptibility = rpaSusceptibilityCalculator->calculateSpinRPASusceptibility(
				kDual,
				{c0_o, a1_o, c1_i, a0_i}
			);
			for(
				unsigned int n = 0;
				n < energies.size();
				n++
			){
				selfEnergyVertex.at(n) += amplitude_i*amplitude_o*spinSusceptibility.at(n);
			}
		}
	}

	vertexTree.add(selfEnergyVertex, resultIndex);

	return selfEnergyVertex;
}

}	//End of namesapce TBTK
