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

/** @file ElectronFluctuationVertex.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Functions.h"
#include "TBTK/InteractionAmplitude.h"
#include "TBTK/Solver/ElectronFluctuationVertex.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

namespace TBTK{
namespace Solver{

ElectronFluctuationVertex::ElectronFluctuationVertex(
	const MomentumSpaceContext &momentumSpaceContext,
	const Property::Susceptibility &susceptibility
//	const Property::Susceptibility &chargeSusceptibility,
//	const Property::Susceptibility &spinSusceptibility
) :
	Communicator(true),
	momentumSpaceContext(momentumSpaceContext),
	susceptibility(susceptibility)
//	chargeSusceptibility(chargeSusceptibility),
//	spinSusceptibility(spinSusceptibility)
{
/*	U = 0.;
	Up = 0.;
	J = 0.;
	Jp = 0.;*/

	numOrbitals = 0;

//	interactionAmplitudesAreGenerated = false;

//	isInitialized = true;
}

/*void ElectronFluctuationVertex::generateInteractionAmplitudes(){
	if(interactionAmplitudesAreGenerated)
		return;

	u1.clear();
	u2.clear();
	u3.clear();

	TBTKAssert(
		numOrbitals != 0,
		"Solver::ElectronFluctuationVertex::generateInteractionAmplitudes()",
		"'numOrbitals' must be non-zero.",
		"Use Solver::ElectronFluctuationVertex::setNumOrbitals() to"
		<< " set the number of orbitals."
	);

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

vector<complex<double>> ElectronFluctuationVertex::calculateSelfEnergyVertexOld(
	const Index &index
){
	TBTKAssert(
		spinSusceptibility.getEnergyType()
		== chargeSusceptibility.getEnergyType(),
		"Solver::ElectronFluctuationVertex::calculateSelfEnergyVertex()",
		"The spin and charge susceptibilities must have the same"
		<< " energy type.",
		""
	);
	unsigned int numEnergies;
	switch(spinSusceptibility.getEnergyType()){
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::Real:
		TBTKAssert(
			spinSusceptibility.getResolution()
			== chargeSusceptibility.getResolution(),
			"Solver::ElectronFluctuationVertex::calculateSelfEnergyVertex()",
			"The spin and charge susceptibilities must have the"
			<< " same energy resolution. The spin susceptibility"
			<< " energy resolution is '"
			<< spinSusceptibility.getResolution() << "' while the"
			<< " charge susceptibility energy resolution is '"
			<< chargeSusceptibility.getResolution() << "'.",
			""
		);
		numEnergies = spinSusceptibility.getResolution();

		break;
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::BosonicMatsubara:
		TBTKAssert(
			spinSusceptibility.getNumMatsubaraEnergies()
			== chargeSusceptibility.getNumMatsubaraEnergies(),
			"Solver::ElectronFluctuationVertex::calculateSelfEnergyVertex()",
			"The spin and charge susceptibilities must have the"
			<< " same number of Matsubara energies. The spin"
			<< " susceptibility has '"
			<< spinSusceptibility.getNumMatsubaraEnergies() << "'"
			<< " Matsubara energies while the charge"
			<< " susceptibility has '"
			<< chargeSusceptibility.getNumMatsubaraEnergies() << "'"
			<< " Matsubara energies.",
			""
		);

		numEnergies = spinSusceptibility.getNumMatsubaraEnergies();
		break;
	default:
		TBTKExit(
			"Solver::ElectronFluctuationVertex::calculateSelfEnergyVertex()",
			"Unknow EnergyType.",
			""
		);
	}

	generateInteractionAmplitudes();

	vector<complex<double>> selfEnergyVertex;
	selfEnergyVertex.reserve(numEnergies);
	for(
		unsigned int n = 0;
		n < numEnergies;
		n++
	){
		selfEnergyVertex.push_back(0.);
	}

	//U_1*\chi_1*U_1
	calculateSelfEnergyVertexMainAlgorithm(
		selfEnergyVertex,
		index,
		chargeSusceptibility,
		u1,
		u1,
		1/2.
	);
	calculateSelfEnergyVertexMainAlgorithm(
		selfEnergyVertex,
		index,
		spinSusceptibility,
		u1,
		u1,
		1/2.
	);

	//U_2*\chi_1*U_2
	calculateSelfEnergyVertexMainAlgorithm(
		selfEnergyVertex,
		index,
		chargeSusceptibility,
		u2,
		u2,
		1/2.
	);
	calculateSelfEnergyVertexMainAlgorithm(
		selfEnergyVertex,
		index,
		spinSusceptibility,
		u2,
		u2,
		1/2.
	);

	//U_1*\chi_2*U_2
	calculateSelfEnergyVertexMainAlgorithm(
		selfEnergyVertex,
		index,
		chargeSusceptibility,
		u1,
		u2,
		1/2.
	);
	calculateSelfEnergyVertexMainAlgorithm(
		selfEnergyVertex,
		index,
		spinSusceptibility,
		u1,
		u2,
		-1/2.
	);

	//U_2*\chi_2*U_1
	calculateSelfEnergyVertexMainAlgorithm(
		selfEnergyVertex,
		index,
		chargeSusceptibility,
		u2,
		u1,
		1/2.
	);
	calculateSelfEnergyVertexMainAlgorithm(
		selfEnergyVertex,
		index,
		spinSusceptibility,
		u2,
		u1,
		-1/2.
	);

	//U_3*\chi_3*U_3
	calculateSelfEnergyVertexMainAlgorithm(
		selfEnergyVertex,
		index,
		spinSusceptibility,
		u3,
		u3,
		1.
	);

	return selfEnergyVertex;
}*/

vector<complex<double>> ElectronFluctuationVertex::calculateSelfEnergyVertex(
	const Index &index
){
	unsigned int numEnergies;
	switch(susceptibility.getEnergyType()){
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::Real:
		numEnergies = susceptibility.getResolution();

		break;
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::BosonicMatsubara:
		numEnergies = susceptibility.getNumMatsubaraEnergies();

		break;
	default:
		TBTKExit(
			"Solver::ElectronFluctuationVertex::calculateSelfEnergyVertex()",
			"Unknow EnergyType.",
			""
		);
	}

	vector<complex<double>> selfEnergyVertex;
	selfEnergyVertex.reserve(numEnergies);
	for(
		unsigned int n = 0;
		n < numEnergies;
		n++
	){
		selfEnergyVertex.push_back(0.);
	}

	calculateSelfEnergyVertexMainAlgorithm(
		selfEnergyVertex,
		index,
		susceptibility,
		leftInteraction,
		rightInteraction,
		multiplier
	);

	return selfEnergyVertex;
}

void ElectronFluctuationVertex::calculateSelfEnergyVertexMainAlgorithm(
	vector<complex<double>> &selfEnergyVertex,
	const Index &index,
	const Property::Susceptibility &susceptibility,
	const vector<InteractionAmplitude> &uLeft,
	const vector<InteractionAmplitude> &uRight,
	double multiplier
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 5,
		"Solver::ElectronFluctuationVertex::calculateSelfEnergyVertexMainAlgorithm()",
		"The Index must be a compound Index with 5 component Indices,"
		<< " but '" << components.size() << "' components supplied.",
		""
	);
	const Index kIndex = components[0];
	const Index intraBlockIndices[4] = {
		components[1],
		components[2],
		components[3],
		components[4]
	};
	for(unsigned int n = 0; n < 4; n++){
		TBTKAssert(
			intraBlockIndices[n].getSize() == 1,
			"Solver::ElectronFluctuationVertex::calculateSelfEnergyVertexMainAlgorithm()",
			"The four last components of the compound Index"
			<< " currently is restricted to have a single"
			<< " subindex, but component '" << n+1 << "' has '"
			<< intraBlockIndices[n].getSize() << "' subindices.",
			"Contact the developer if support for more general"
			<< " Indices is required."
		);
		//TODO
		//Rewrite code below to not depend on the intraBlockIndices
		//having a single subindex each. The remove this assert
		//statement.
	}

	for(unsigned int in = 0; in < uLeft.size(); in++){
		const InteractionAmplitude &incommingAmplitude = uLeft.at(in);
		complex<double> amplitude_i = incommingAmplitude.getAmplitude();
		int c0_i = incommingAmplitude.getCreationOperatorIndex(0).at(0);
		int c1_i = incommingAmplitude.getCreationOperatorIndex(1).at(0);
		int a0_i = incommingAmplitude.getAnnihilationOperatorIndex(0).at(0);
		int a1_i = incommingAmplitude.getAnnihilationOperatorIndex(1).at(0);

		if(
			a1_i != intraBlockIndices[3][0]
			|| c0_i != intraBlockIndices[2][0]
			|| abs(amplitude_i) < 1e-10
		){
			continue;
		}

		for(unsigned int out = 0; out < uRight.size(); out++){
			const InteractionAmplitude &outgoingAmplitude = uRight.at(out);
			complex<double> amplitude_o = outgoingAmplitude.getAmplitude();
			int c0_o = outgoingAmplitude.getCreationOperatorIndex(0).at(0);
			int c1_o = outgoingAmplitude.getCreationOperatorIndex(1).at(0);
			int a0_o = outgoingAmplitude.getAnnihilationOperatorIndex(0).at(0);
			int a1_o = outgoingAmplitude.getAnnihilationOperatorIndex(1).at(0);

			if(
				a0_o != intraBlockIndices[0][0]
				|| c1_o != intraBlockIndices[1][0]
				|| abs(amplitude_o) < 1e-10
			){
				continue;
			}

			const vector<complex<double>> &susceptibilityData
				= susceptibility.getData();
			unsigned int offsetChargeSusceptibility
				= susceptibility.getOffset({
					kIndex,
					{c0_o},
					{a1_o},
					{c1_i},
					{a0_i}
				});

			for(
				unsigned int n = 0;
				n < selfEnergyVertex.size();
				n++
			){
				selfEnergyVertex.at(n) += amplitude_i*amplitude_o*(
					susceptibilityData[
						offsetChargeSusceptibility + n
					]
				)*multiplier;
			}
		}
	}
}

}	//End of namespace Solver
}	//End of namesapce TBTK
