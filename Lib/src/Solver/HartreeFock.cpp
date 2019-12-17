/* Copyright 2019 Kristofer Björnson
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

/** @file HartreeFock.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/AbstractState.h"
#include "TBTK/HartreeFockPotentialOperator.h"
#include "TBTK/KineticOperator.h"
#include "TBTK/MultiCounter.h"
#include "TBTK/NuclearPotentialOperator.h"
#include "TBTK/Property/WaveFunctions.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Solver/HartreeFock.h"
#include "TBTK/UnitHandler.h"

using namespace std;

namespace TBTK{
namespace Solver{

HartreeFock::HartreeFock() : selfConsistencyCallback(*this){
	occupationNumber = 0;
	totalEnergy = 0;
}

HartreeFock::~HartreeFock(){
}

void HartreeFock::run(){
	const BasisStateSet &basisStateSet = getModel().getBasisStateSet();
	for(
		BasisStateSet::ConstIterator iterator = basisStateSet.cbegin();
		iterator != basisStateSet.cend();
		++iterator
	){
		basisStates.push_back(&(*iterator));
	}

	TBTKAssert(
		(int)basisStates.size() == getModel().getBasisSize(),
		"Solver::HartreeFock::run()",
		"The models basis size is different from the number of basis"
		<< " functions.",
		"Ensure that the model has been created by adding basis"
		<< " functions to the model, followed by calls to"
		<< " model.generateHoppingAmplitudeSet(),"
		<< " model.generateOverlapAmplitudeSet(),"
		<< " and model.construct()."
	);

	densityMatrix = Matrix<complex<double>>(
		basisStates.size(),
		basisStates.size()
	);

	setSelfConsistencyCallback(selfConsistencyCallback);

	Diagonalizer::run();
}

void HartreeFock::calculateTotalEnergy(){
	complex<double> complexEnergy = 0;

	unsigned int basisSize = basisStates.size();
	MultiCounter<unsigned int> counter(
		{0, 0},
		{basisSize, basisSize},
		{1, 1}
	);
	for(counter.reset(); !counter.done(); ++counter){
		const AbstractState &braState = *basisStates[counter[0]];
		const AbstractState &ketState = *basisStates[counter[1]];

		//Kinetic term.
		complexEnergy += densityMatrix.at(
			counter[0],
			counter[1]
		)*braState.getMatrixElement(
			ketState,
			KineticOperator(
				UnitHandler::getConstantInNaturalUnits("m_e")
			)
		);

		//Nuclear potential term.
		for(
			unsigned int n = 0;
			n < nuclearCenters.size();
			n++
		){
			complexEnergy += densityMatrix.at(
				counter[0],
				counter[1]
			)*braState.getMatrixElement(
				ketState,
				NuclearPotentialOperator(
					nuclearCenters[n],
					nuclearCenters[n].getPosition()
				)
			);
		}
	}

	//Hartree-Fock term.
	counter = MultiCounter<unsigned int>(
		{0, 0, 0, 0},
		{basisSize, basisSize, basisSize, basisSize},
		{1, 1, 1, 1}
	);
	for(counter.reset(); !counter.done(); ++counter){
		const AbstractState &braState = *basisStates[counter[0]];
		const AbstractState &ketState = *basisStates[counter[1]];

		complexEnergy += densityMatrix.at(
			counter[0],
			counter[1]
		)*densityMatrix.at(
			counter[2],
			counter[3]
		)*braState.getMatrixElement(
			ketState,
			HartreeFockPotentialOperator(
				*basisStates[counter[2]],
				*basisStates[counter[3]]
			)
		)/2.;
	}

	//Intra-nuclear potential energy.
	double e = UnitHandler::getConstantInNaturalUnits("e");
	double epsilon_0 = UnitHandler::getConstantInNaturalUnits("epsilon_0");
	double prefactor = pow(e, 2)/(4*M_PI*epsilon_0);
	for(unsigned int m = 0; m < nuclearCenters.size(); m++){
		for(unsigned int n = 0; n < nuclearCenters.size(); n++){
			if(m == n)
				continue;

			double z0 = nuclearCenters[m].getAtomicNumber();
			double z1 = nuclearCenters[n].getAtomicNumber();
			double r = (
				nuclearCenters[m].getPosition()
				- nuclearCenters[n].getPosition()
			).norm();

			complexEnergy += (1/2.)*prefactor*z0*z1/r;
		}
	}

	totalEnergy = real(complexEnergy);
}

HartreeFock::Callbacks::Callbacks(){
	solver = nullptr;
}

complex<double> HartreeFock::Callbacks::getHoppingAmplitude(
	const Index &to,
	const Index &from
) const{
	TBTKAssert(
		solver != nullptr,
		"Solver::HartreeFock::Callbacks::getHoppingAmplitude()",
		"Solver not set.",
		"Use Solver::HartreeFock::Callbacks::setSolver() to set the"
		<< " Solver that the Callbacks is associate with."
	);

	complex<double> amplitude = 0;
	const Model &model = solver->getModel();
	const Matrix<complex<double>> &densityMatrix
		= solver->getDensityMatrix();
	const vector<PositionedAtom> nuclearCenters
		= solver->getNuclearCenters();

	const BasisStateSet &basisStateSet = model.getBasisStateSet();
	const AbstractState &braState = basisStateSet.get(to);
	const AbstractState &ketState = basisStateSet.get(from);

	//Kinetic term.
	amplitude += braState.getMatrixElement(
		ketState,
		KineticOperator(UnitHandler::getConstantInNaturalUnits("m_e"))
	);

	//Hartree-Fock potential.
	const vector<const AbstractState*> &basisStates = solver->basisStates;
	for(unsigned int m = 0; m< basisStates.size(); m++){
		for(unsigned int n = 0; n < basisStates.size(); n++){
			amplitude += densityMatrix.at(
				m,
				n
			)*braState.getMatrixElement(
				ketState,
				HartreeFockPotentialOperator(
					*basisStates[m],
					*basisStates[n]
				)
			);
		}
	}

	//Nuclear potential term.
	for(unsigned int n = 0; n < nuclearCenters.size(); n++){
		amplitude += braState.getMatrixElement(
			ketState,
			NuclearPotentialOperator(
				nuclearCenters[n],
				nuclearCenters[n].getPosition()
			)
		);
	}

	return amplitude;
}

complex<double> HartreeFock::Callbacks::getOverlapAmplitude(
	const Index &bra,
	const Index &ket
) const{
	TBTKAssert(
		solver != nullptr,
		"Solver::HartreeFock::Callbacks::getOverlapAmplitude()",
		"Solver not set.",
		"Use Solver::HartreeFock::Callbacks::setSolver() to set the"
		<< " Solver that the Callbacks is associate with."
	);

	const BasisStateSet &basisStateSet
		= solver->getModel().getBasisStateSet();

	return basisStateSet.get(bra).getOverlap(basisStateSet.get(ket));
}

HartreeFock::SelfConsistencyCallback::SelfConsistencyCallback(
	HartreeFock &solver
) : solver(solver){
}

bool HartreeFock::SelfConsistencyCallback::selfConsistencyCallback(
	Diagonalizer &diagonalizer
){
	HartreeFock &solver = (HartreeFock&)diagonalizer;

	PropertyExtractor::Diagonalizer propertyExtractor(diagonalizer);
	vector<Index> basisIndices;
	for(unsigned int n = 0; n < solver.basisStates.size(); n++)
		basisIndices.push_back(solver.basisStates[n]->getIndex());
	Property::WaveFunctions waveFunctions
		= propertyExtractor.calculateWaveFunctions(
			basisIndices,
			{IDX_ALL}
		);

	solver.densityMatrix = Matrix<complex<double>>(
		basisIndices.size(),
		basisIndices.size()
	);
	for(unsigned int m = 0; m < basisIndices.size(); m++){
		for(unsigned int n = 0; n < basisIndices.size(); n++){
			solver.densityMatrix.at(m, n) = 0;
			for(
				unsigned int c = 0;
				c < solver.occupationNumber;
				c++
			){
				solver.densityMatrix.at(m, n) += conj(
					waveFunctions(
						basisIndices[m],
						c
					)
				)*waveFunctions(
					basisIndices[n],
					c
				);
			}
		}
	}

	double oldTotalEnergy = solver.getTotalEnergy();
	solver.calculateTotalEnergy();
//	solver.totalEnergy = solver.getTotalEnergy();

	if(abs(solver.totalEnergy - oldTotalEnergy) < 1e-6)
		return true;
	else
		return false;
}

};	//End of namespace Solver
};	//End of namespace TBTK
