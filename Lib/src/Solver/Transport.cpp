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

DynamicTypeInformation Transport::dynamicTypeInformation(
	"Solver::Transport",
	{&Solver::dynamicTypeInformation}
);

Transport::Transport(
) :
	Communicator(false),
	energyRange(-1, 1, 1000)
{
}

double Transport::calculateCurrent(
	unsigned int lead0/*,
	unsigned int lead1*/
){
	TBTKAssert(
		lead0 < leads.size(),
		"Solver::Transport::calculateTransmissionRate()",
		"'lead0' must be a number between 0 and one less than the"
		<< " number of leads, but 'lead0=" << lead0 << "' and the"
		<< " number of leads is '" << leads.size() << "'.",
		""
	);
/*	TBTKAssert(
		lead1 < leads.size(),
		"Solver::Transport::calculateTransmissionRate()",
		"'lead1' must be a number between 0 and one less than the"
		<< " number of leads, but 'lead1=" << lead1 << "' and the"
		<< " number of leads is '" << leads.size() << "'.",
		""
	);*/

	calculateGreensFunction();
	calculateInteractingGreensFunction();
	calculateBroadenings();
	calculateInscatterings();
	calculateFullInscattering();
	calculateCorrelationFunction();
	calculateSpectralFunction();
	calculateEnergyResolvedCurrents();
	calculateCurrents();

/*	Greens solver;
	solver.setModel(getModel());
	solver.setGreensFunction(interactingGreensFunction);
	return solver.calculateTransmissionRate(
			expandSelfEnergyIndexRange(leads[lead0].selfEnergy),
			expandSelfEnergyIndexRange(leads[lead1].selfEnergy)
		);*/

	return leads[lead0].current;
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
				) = conj(lead.selfEnergy(index, energy));
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
	Timer::tick("Expand self energy index range");
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

	Timer::tock();

	return expandedSelfEnergy;
}

void Transport::calculateCorrelationFunction(){
	Timer::tick("Calculate correlation function");
	vector<SparseMatrix<complex<double>>> G
		= greensFunction.toSparseMatrices(getModel());
	vector<SparseMatrix<complex<double>>> sigmaIn
		= fullInscattering.toSparseMatrices(getModel());
	vector<SparseMatrix<complex<double>>> GDagger;
	for(unsigned int n = 0; n < G.size(); n++)
		GDagger.push_back(G[n].hermitianConjugate());

	correlationFunction = Property::EnergyResolvedProperty<complex<double>>(
		greensFunction.getIndexDescriptor().getIndexTree(),
		greensFunction.getLowerBound(),
		greensFunction.getUpperBound(),
		greensFunction.getResolution()
	);
	for(unsigned int energy = 0; energy < greensFunction.getResolution(); energy++){
		SparseMatrix<complex<double>> product
			= G[energy]*sigmaIn[energy]*GDagger[energy];

		product.setStorageFormat(
			SparseMatrix<complex<double>>::StorageFormat::CSC
		);
		const unsigned int *cscColumnPointers
			= product.getCSCColumnPointers();
		const unsigned int *cscRows = product.getCSCRows();
		const complex<double> *values = product.getCSCValues();

		for(
			unsigned int column = 0;
			column < product.getNumColumns();
			column++
		){
			Index columnIndex = getModel().getHoppingAmplitudeSet(
			).getPhysicalIndex(column);
			for(
				unsigned int n = cscColumnPointers[column];
				n < cscColumnPointers[column+1];
				n++
			){
				unsigned int row = cscRows[n];
				Index rowIndex
					= getModel().getHoppingAmplitudeSet(
					).getPhysicalIndex(row);
				correlationFunction(
					{rowIndex, columnIndex},
					energy
				) = values[n];
			}
		}
	}

	Timer::tock();
}

void Transport::calculateSpectralFunction(){
	Timer::tick("Calculate spectral function");
	Greens solver;
	solver.setGreensFunction(interactingGreensFunction);
	spectralFunction = solver.calculateSpectralFunction();
	Timer::tock();
}

void Transport::calculateEnergyResolvedCurrents(){
	Timer::tick("Calculate energy-resolved current");
	double hbar = UnitHandler::getConstantInNaturalUnits("hbar");
	double e = UnitHandler::getConstantInNaturalUnits("e");
	for(auto &lead : leads){
		lead.energyResolvedCurrent
			= Property::EnergyResolvedProperty<double>(
				greensFunction.getLowerBound(),
				greensFunction.getUpperBound(),
				greensFunction.getResolution()
			);

		vector<SparseMatrix<complex<double>>> sigmaIn
			= lead.inscattering.toSparseMatrices(getModel());
		vector<SparseMatrix<complex<double>>> A
			= spectralFunction.toSparseMatrices(getModel());
		TBTKAssert(
			sigmaIn.size() == A.size(),
			"Solver::Transport::calculateEnergyResolvedCurrents()",
			"Incompatible energy ranges for the inscattering and"
			<< " the spectral function.",
			"This should never happen, contact the developer."
		);
		for(unsigned int n = 0; n < sigmaIn.size(); n++){
			lead.energyResolvedCurrent(n)
				= real((sigmaIn[n]*A[n]).trace());
		}

		vector<SparseMatrix<complex<double>>> Gamma
			= lead.broadening.toSparseMatrices(getModel());
		vector<SparseMatrix<complex<double>>> G
			= correlationFunction.toSparseMatrices(getModel());
		TBTKAssert(
			sigmaIn.size() == Gamma.size()
			&& Gamma.size() == G.size(),
			"Solver::Transport::calculateEnergyResolvedCurrents()",
			"Incompatible energy ranges for the inscattering,"
			<< " broadening, and correlation function.",
			"This should never happen, contact the developer."
		);
		for(unsigned int n = 0; n < Gamma.size(); n++){
			lead.energyResolvedCurrent(n)
				-= real((Gamma[n]*G[n]).trace());
		}

		for(unsigned int n = 0; n < Gamma.size(); n++)
			lead.energyResolvedCurrent(n) *= e/(2*M_PI*hbar);
	}
	Timer::tock();
}

void Transport::calculateCurrents(){
	Timer::tick("Calculate currents");
	for(auto &lead : leads){
		lead.current = 0;
		double dE = lead.energyResolvedCurrent.getDeltaE();
		for(
			unsigned int n = 0;
			n < lead.energyResolvedCurrent.getResolution();
			n++
		){
			lead.current += lead.energyResolvedCurrent(n)*dE;
		}
	}
	Timer::tock();
}

};	//End of namespace Solver
};	//End of namespace TBTK
