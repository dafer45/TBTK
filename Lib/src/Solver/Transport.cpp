/* Copyright 2020 Kristofer Björnson
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

/** @file Transport.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Functions.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Solver/Greens.h"
#include "TBTK/Solver/Transport.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/Timer.h"

using namespace std;

namespace TBTK{
namespace Solver{

Transport::Transport(
) :
	Communicator(false),
	energyRange(-1, 1, 1000)
{
}

Property::TransmissionRate Transport::calculateTransmissionRate(
	unsigned int lead0,
	unsigned int lead1
){
	TBTKAssert(
		lead0 < leads.size(),
		"Solver::Transport::calculateTransmissionRate()",
		"'lead0' must be a number between 0 and one less than the"
		<< " number of leads, but 'lead0=" << lead0 << "' and the"
		<< " number of leads is '" << leads.size() << "'.",
		""
	);
	TBTKAssert(
		lead1 < leads.size(),
		"Solver::Transport::calculateTransmissionRate()",
		"'lead1' must be a number between 0 and one less than the"
		<< " number of leads, but 'lead1=" << lead1 << "' and the"
		<< " number of leads is '" << leads.size() << "'.",
		""
	);

	calculateGreensFunction();
	calculateInteractingGreensFunction();
	calculateBroadenings();
	calculateInscatterings();
	calculateFullInscattering();

	Greens solver;
	solver.setModel(getModel());
	solver.setGreensFunction(interactingGreensFunction);
	return solver.calculateTransmissionRate(
			expandSelfEnergyIndexRange(leads[lead0].selfEnergy),
			expandSelfEnergyIndexRange(leads[lead1].selfEnergy)
		);
}

void Transport::calculateGreensFunction(){
	Timer::tick("Calculate Green's function");
	Diagonalizer solver;
	solver.setModel(getModel());
	solver.run();

	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	Property::EigenValues eigenValues = propertyExtractor.getEigenValues();
	propertyExtractor.setEnergyInfinitesimal(1e-10);
	propertyExtractor.setEnergyWindow(
		energyRange[0],
		energyRange[energyRange.getResolution()-1],
		energyRange.getResolution()
	);

	vector<Index> indices;
	const IndexTree &indexTree
		= getModel().getHoppingAmplitudeSet().getIndexTree();
	for(auto index0 : indexTree)
		for(auto index1 : indexTree)
			indices.push_back({index0, index1});

	greensFunction = propertyExtractor.calculateGreensFunction(indices);
	Timer::tock();
}

void Transport::calculateInteractingGreensFunction(){
	Timer::tick("Calculate interacting Green's function");
	calculateFullSelfEnergy();

	Greens solver;
	solver.setModel(getModel());
	solver.setGreensFunction(greensFunction);
	interactingGreensFunction
		= solver.calculateInteractingGreensFunction(fullSelfEnergy);
	Timer::tock();
}

void Transport::calculateFullSelfEnergy(){
	Timer::tick("Calculate full self-energy");
	fullSelfEnergy = Property::SelfEnergy(
		greensFunction.getIndexDescriptor().getIndexTree(),
		greensFunction.getLowerBound(),
		greensFunction.getUpperBound(),
		greensFunction.getResolution()
	);
	for(auto &lead : leads){
		for(auto index : lead.selfEnergy.getIndexDescriptor().getIndexTree()){
			TBTKAssert(
				fullSelfEnergy.contains(index),
				"Solver::Transport::calculateFullSelfEnergy()",
				"Encountered the Index '" << index << "' in"
				<< " one of the lead self-energies, but the"
				<< " Index is not contained in the Green's"
				<< " function.",
				""
			);
			TBTKAssert(
				lead.selfEnergy.getResolution()
					== fullSelfEnergy.getResolution(),
				"Solver::Transport::calculateFullSelfEnergy()",
				"One of the lead self-energies has a different"
				<< " energy resolution than the Green's"
				<< " function.",
				""
			);
			for(
				unsigned int energy = 0;
				energy < fullSelfEnergy.getResolution();
				energy++
			){
				fullSelfEnergy(index, energy)
					+= lead.selfEnergy(index, energy);
			}
		}
	}
	Timer::tock();
}

void Transport::calculateBroadenings(){
	Timer::tick("Calculate broadening");
	complex<double> i(0, 1);
	for(auto &lead : leads){
		const IndexTree &indexTree
			= lead.selfEnergy.getIndexDescriptor().getIndexTree();
		Property::SelfEnergy hermitianConjugate(
			indexTree,
			lead.selfEnergy.getLowerBound(),
			lead.selfEnergy.getUpperBound(),
			lead.selfEnergy.getResolution()
		);
		for(auto index : indexTree){
			vector<Index> components = index.split();
			for(
				unsigned int energy = 0;
				energy < lead.selfEnergy.getResolution();
				energy++
			){
				hermitianConjugate(
					{components[1], components[0]},
					energy
				) = lead.selfEnergy(index, energy);
			}
		}

		lead.broadening = i*(lead.selfEnergy - hermitianConjugate);
	}
	Timer::tock();
}

void Transport::calculateInscatterings(){
	Timer::tick("Calculate inscattering");
	for(auto &lead : leads){
		lead.inscattering = lead.broadening;
		for(
			auto index :
				lead.inscattering.getIndexDescriptor(
				).getIndexTree()
		){
			for(
				unsigned int e = 0;
				e < lead.inscattering.getResolution();
				e++
			){
				lead.inscattering(index, e)
					*= Functions::fermiDiracDistribution(
						lead.broadening.getEnergy(e),
						lead.chemicalPotential,
						lead.temperature
					);
			}
		}
	}
	Timer::tock();
}

void Transport::calculateFullInscattering(){
	Timer::tick("Calculate full inscattering");
	fullInscattering = Property::SelfEnergy(
		greensFunction.getIndexDescriptor().getIndexTree(),
		greensFunction.getLowerBound(),
		greensFunction.getUpperBound(),
		greensFunction.getResolution()
	);
//	bool first = true;
	for(auto &lead : leads){
/*		if(first){
			fullInscattering = lead.inscattering;
			first = false;
		}
		else{
			fullInscattering += lead.inscattering;
		}*/
		for(auto index : lead.inscattering.getIndexDescriptor().getIndexTree()){
			TBTKAssert(
				fullInscattering.contains(index),
				"Solver::Transport::calculateFullInscattering()",
				"Encountered the Index '" << index << "' in"
				<< " one of the lnscatterings, but the Index is"
				<< " not contained in the Green's function.",
				""
			);
			TBTKAssert(
				lead.selfEnergy.getResolution()
					== fullSelfEnergy.getResolution(),
				"Solver::Transport::calculateFullInscattering()",
				"One of the inscatterings has a different"
				<< " energy resolution than the Green's"
				<< " function.",
				""
			);
			for(
				unsigned int energy = 0;
				energy < fullInscattering.getResolution();
				energy++
			){
				fullInscattering(index, energy)
					+= lead.inscattering(index, energy);
			}
		}
	}
	Timer::tock();
}

Property::SelfEnergy Transport::expandSelfEnergyIndexRange(
	const Property::SelfEnergy &selfEnergy
) const{
	Property::SelfEnergy expandedSelfEnergy(
		greensFunction.getIndexDescriptor().getIndexTree(),
		greensFunction.getLowerBound(),
		greensFunction.getUpperBound(),
		greensFunction.getResolution()
	);
	for(auto &index : selfEnergy.getIndexDescriptor().getIndexTree()){
		for(
			unsigned int energy = 0;
			energy < selfEnergy.getResolution();
			energy++
		){
			expandedSelfEnergy(index, energy)
				= selfEnergy(index, energy);
		}
	}

	return expandedSelfEnergy;
}

};	//End of namespace Solver
};	//End of namespace TBTK
