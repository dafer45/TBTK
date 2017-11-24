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

#include "Functions.h"
#include "InteractionAmplitude.h"
#include "RPA/SelfEnergyCalculator.h"
#include "UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

const complex<double> i(0, 1);

namespace TBTK{

SelfEnergyCalculator::SelfEnergyCalculator(
	const MomentumSpaceContext &momentumSpaceContext,
	unsigned int numWorkers
){
	TBTKAssert(
		numWorkers > 0,
		"SelfEnergyCalculator::SelfEnergyCalculator()",
		"'numWorkers' must be larger than zero.",
		""
	);

	isInitialized = false;

	kMinusQLookupTable = nullptr;

	U = 0.;
	Up = 0.;
	J = 0.;
	Jp = 0.;

	susceptibilityCalculators.push_back(
		new SusceptibilityCalculator(momentumSpaceContext)
	);
	for(unsigned int n = 1; n < numWorkers; n++){
		susceptibilityCalculators.push_back(
			susceptibilityCalculators[0]->createSlave()
		);
	}

	for(unsigned int n = 0; n < numWorkers; n++){
		selfEnergyVertexTrees.push_back(
			IndexedDataTree<SerializeableVector<complex<double>>>()
		);
	}

	interactionAmplitudesAreGenerated = false;
}

SelfEnergyCalculator::~SelfEnergyCalculator(){
	if(kMinusQLookupTable != nullptr)
		delete [] kMinusQLookupTable;
}

void SelfEnergyCalculator::init(){
	TBTKAssert(
		numSummationEnergies%2 == 1,
		"SelfEnergyCalculator::int()",
		"The number of summation energies must be an odd number.",
		"Use SelfEnergyCalculator::setNumSummationEnergies() to set"
		<< " the number of summation energies."
	);

	if(kMinusQLookupTable != nullptr){
		delete [] kMinusQLookupTable;
		kMinusQLookupTable = nullptr;
	}

	//Calculate kT
	double temperature = UnitHandler::convertTemperatureNtB(
		susceptibilityCalculators[0]->getMomentumSpaceContext(
		).getModel().getTemperature()
	);
	double kT = UnitHandler::getK_BB()*temperature;

	//Initialize summation energies
	for(
		int n = -(int)numSummationEnergies/2;
		n <= (int)numSummationEnergies/2;
		n++
	){
		summationEnergies.push_back(i*M_PI*2.*(double)(n)*kT);
	}
	for(unsigned int n = 0; n < susceptibilityCalculators.size(); n++){
		susceptibilityCalculators[n]->setSusceptibilityEnergies(
			summationEnergies
		);
		susceptibilityCalculators[n]->setSusceptibilityEnergyType(
			SusceptibilityCalculator::EnergyType::Imaginary
		);
		susceptibilityCalculators[n]->setSusceptibilityEnergiesAreInversionSymmetric(
			true
		);
	}

/*	Timer::tick("Precompute");
	susceptibilityCalculator.precompute();
	Timer::tock();*/

	isInitialized = true;
}

void SelfEnergyCalculator::generateKMinusQLookupTable(){
	if(kMinusQLookupTable != nullptr)
		return;

	Timer::tick("Calculate k-q lookup table.");
	const MomentumSpaceContext &momentumSpaceContext = susceptibilityCalculators[0]->getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();
	const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
	const Model &model = momentumSpaceContext.getModel();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	kMinusQLookupTable = new int[mesh.size()*mesh.size()];

	string cacheName = "cache/kMinusQLookupTable";
	for(
		unsigned int n = 0;
		n < numMeshPoints.size();
		n++
	){
		cacheName += "_" + to_string(numMeshPoints.at(n));
	}
	ifstream fin(cacheName);
	if(fin){
		unsigned int counter = 0;
		int value;
		while(fin >> value){
			TBTKAssert(
				counter < mesh.size()*mesh.size(),
				"SelfEnergyCalculator::generateKMinusQLookupTable()",
				"Found cache file '" << cacheName << "',"
				<< " but the cache is too large.",
				"Clear the cache to recalculate"
				<< " kMinusQLookupTable."
			);
			kMinusQLookupTable[counter] = value;
			counter++;
		}
		fin.close();

		TBTKAssert(
			counter == mesh.size()*mesh.size(),
			"SelfEnergyCalculator::generateKMinusQLookupTable()",
			"Found cache file '" << cacheName << "',"
			<< " but the cache is too small.",
			"Clear the cache to recalculate kMinusQLookupTable."
		);

		Timer::tock();

		return;
	}

	#pragma omp parallel for
	for(unsigned int k = 0; k < mesh.size(); k++){
		const vector<double> &K = mesh.at(k);
		for(unsigned int q = 0; q < mesh.size(); q++){
			const vector<double> &Q = mesh.at(q);

			vector<double> kMinusQ;
			for(unsigned int n = 0; n < K.size(); n++)
				kMinusQ.push_back(K.at(n) - Q.at(n));

			Index qIndex = brillouinZone.getMinorCellIndex(
				Q,
				numMeshPoints
			);
			int qLinearIndex = model.getHoppingAmplitudeSet()->getFirstIndexInBlock(
				qIndex
			);
			Index kMinusQIndex = brillouinZone.getMinorCellIndex(
				kMinusQ,
				numMeshPoints
			);
			kMinusQLookupTable[
				k*mesh.size() + qLinearIndex/numOrbitals
			] = model.getHoppingAmplitudeSet()->getFirstIndexInBlock(
				kMinusQIndex
			);
		}
	}

	ofstream fout(cacheName);
	if(fout)
		for(unsigned int n = 0; n < mesh.size()*mesh.size(); n++)
			fout << kMinusQLookupTable[n] << "\n";

	Timer::tock();
}

template<>
inline int SelfEnergyCalculator::getKMinusQLinearIndex<false>(
	unsigned int meshIndex,
	const vector<double> &k,
	int kLinearIndex
) const{
	const MomentumSpaceContext &momentumSpaceContext = susceptibilityCalculators[0]->getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	Index kMinusQIndex = momentumSpaceContext.getBrillouinZone().getMinorCellIndex(
		{mesh[meshIndex][0] + k[0], mesh[meshIndex][1] + k[1]},
		momentumSpaceContext.getNumMeshPoints()
	);
	return momentumSpaceContext.getModel().getHoppingAmplitudeSet()->getFirstIndexInBlock(
		kMinusQIndex
	);
}

template<>
inline int SelfEnergyCalculator::getKMinusQLinearIndex<true>(
	unsigned int meshIndex,
	const vector<double> &k,
	int kLinearIndex
) const{
	const MomentumSpaceContext &momentumSpaceContext = susceptibilityCalculators[0]->getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();
	return kMinusQLookupTable[
		meshIndex*mesh.size() + kLinearIndex/numOrbitals
	];
}

extern "C" {
	void zgetrf_(
		int* M,
		int *N,
		complex<double> *A,
		int *lda,
		int *ipiv,
		int *info
	);
	void zgetri_(
		int *N,
		complex<double> *A,
		int *lda,
		int *ipiv,
		complex<double> *work,
		int *lwork,
		int *info
	);
}

void SelfEnergyCalculator::invertMatrix(
	complex<double> *matrix,
	unsigned int dimensions
){
	int numRows = dimensions;
	int numCols = dimensions;

	int *ipiv = new int[min(numRows, numCols)];
	int lwork = numCols*numCols;
	complex<double> *work = new complex<double>[lwork];
	int info;

	zgetrf_(&numRows, &numCols, matrix, &numRows, ipiv, &info);
	zgetri_(&numRows, matrix, &numRows, ipiv, work, &lwork, &info);

	delete [] ipiv;
	delete [] work;
}

void SelfEnergyCalculator::multiplyMatrices(
	complex<double> *matrix1,
	complex<double> *matrix2,
	complex<double> *result,
	unsigned int dimensions
){
	for(unsigned int n = 0; n < dimensions*dimensions; n++)
		result[n] = 0.;

	for(unsigned int row = 0; row < dimensions; row++)
		for(unsigned int col = 0; col < dimensions; col++)
			for(unsigned int n = 0; n < dimensions; n++)
				result[dimensions*col + row] += matrix1[dimensions*n + row]*matrix2[dimensions*col + n];
}

void SelfEnergyCalculator::printMatrix(complex<double> *matrix, unsigned int dimension){
	for(unsigned int r = 0; r < dimension; r++){
		for(unsigned int c = 0; c < dimension; c++){
			Streams::out << setw(20) << matrix[dimension*c + r];
		}
		Streams::out << "\n";
	}
	Streams::out << "\n";
}

void SelfEnergyCalculator::generateInteractionAmplitudes(){
	if(interactionAmplitudesAreGenerated)
		return;

	u1.clear();
	u2.clear();
	u3.clear();

	const MomentumSpaceContext &momentumSpaceContext = susceptibilityCalculators[0]->getMomentumSpaceContext();
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

vector<complex<double>> SelfEnergyCalculator::calculateSelfEnergyVertex(
	const vector<double> &k,
	const vector<int> &orbitalIndices,
	unsigned int worker
){
	TBTKAssert(
		isInitialized,
		"SelfEnergyCalculator::calculateSelfEnergyVertex()",
		"SelfEnergyCalculator not yet initialized.",
		"Use SelfEnergyCalculator::init() to initialize the"
		<< " SelfEnergyCalculator."
	);
	TBTKAssert(
		orbitalIndices.size() == 4,
		"calculateSelfEnergyVertex()",
		"Two orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);
	TBTKAssert(
		worker < selfEnergyVertexTrees.size(),
		"SelfEnergyCalculator::calculateSelfEnergyVertex()",
		"'worker' must be smaller than 'numWorkers' specified in the"
		" constructor (defult value for 'numWorkers' is 1).",
		""
	);

	const MomentumSpaceContext &momentumSpaceContext = susceptibilityCalculators[0]->getMomentumSpaceContext();
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

	SerializeableVector<complex<double>> result;
	if(selfEnergyVertexTrees[worker].get(result, resultIndex))
		return result;


	DualIndex kDual(kIndex, k);

//	generateInteractionAmplitudes();

	vector<complex<double>> selfEnergyVertex;
	selfEnergyVertex.reserve(numSummationEnergies);
	for(
		unsigned int n = 0;
		n < numSummationEnergies;
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
/*			a1_i != orbitalIndices.at(3)
			|| c0_i != orbitalIndices.at(0)*/
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
/*				a0_o != orbitalIndices.at(2)
				|| c1_o != orbitalIndices.at(1)*/
				a0_o != orbitalIndices.at(0)
				|| c1_o != orbitalIndices.at(1)
				|| abs(amplitude_o) < 1e-10
			){
				continue;
			}

			vector<complex<double>> chargeSusceptibility = susceptibilityCalculators[worker]->calculateChargeRPASusceptibility(
				kDual,
//				{c1_i, a0_i, c0_o, a1_o}
				{c0_o, a1_o, c1_i, a0_i}
			);
			vector<complex<double>> spinSusceptibility = susceptibilityCalculators[worker]->calculateSpinRPASusceptibility(
				kDual,
//				{c1_i, a0_i, c0_o, a1_o}
				{c0_o, a1_o, c1_i, a0_i}
			);
			for(
				unsigned int n = 0;
				n < numSummationEnergies;
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
/*			a1_i != orbitalIndices.at(3)
			|| c0_i != orbitalIndices.at(0)*/
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
/*				a0_o != orbitalIndices.at(2)
				|| c1_o != orbitalIndices.at(1)*/
				a0_o != orbitalIndices.at(0)
				|| c1_o != orbitalIndices.at(1)
				|| abs(amplitude_o) < 1e-10
			){
				continue;
			}

			vector<complex<double>> chargeSusceptibility = susceptibilityCalculators[worker]->calculateChargeRPASusceptibility(
				kDual,
//				{c1_i, a0_i, c0_o, a1_o}
				{c0_o, a1_o, c1_i, a0_i}
			);
			vector<complex<double>> spinSusceptibility = susceptibilityCalculators[worker]->calculateSpinRPASusceptibility(
				kDual,
//				{c1_i, a0_i, c0_o, a1_o}
				{c0_o, a1_o, c1_i, a0_i}
			);
			for(
				unsigned int n = 0;
				n < numSummationEnergies;
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
/*			a1_i != orbitalIndices.at(3)
			|| c0_i != orbitalIndices.at(0)*/
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
/*				a0_o != orbitalIndices.at(2)
				|| c1_o != orbitalIndices.at(1)*/
				a0_o != orbitalIndices.at(0)
				|| c1_o != orbitalIndices.at(1)
				|| abs(amplitude_o) < 1e-10
			){
				continue;
			}

			vector<complex<double>> chargeSusceptibility = susceptibilityCalculators[worker]->calculateChargeRPASusceptibility(
				kDual,
//				{c1_i, a0_i, c0_o, a1_o}
				{c0_o, a1_o, c1_i, a0_i}
			);
			vector<complex<double>> spinSusceptibility = susceptibilityCalculators[worker]->calculateSpinRPASusceptibility(
				kDual,
//				{c1_i, a0_i, c0_o, a1_o}
				{c0_o, a1_o, c1_i, a0_i}
			);
			for(
				unsigned int n = 0;
				n < numSummationEnergies;
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
/*			a1_i != orbitalIndices.at(3)
			|| c0_i != orbitalIndices.at(0)*/
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
/*				a0_o != orbitalIndices.at(2)
				|| c1_o != orbitalIndices.at(1)*/
				a0_o != orbitalIndices.at(0)
				|| c1_o != orbitalIndices.at(1)
				|| abs(amplitude_o) < 1e-10
			){
				continue;
			}

			vector<complex<double>> chargeSusceptibility = susceptibilityCalculators[worker]->calculateChargeRPASusceptibility(
				kDual,
//				{c1_i, a0_i, c0_o, a1_o}
				{c0_o, a1_o, c1_i, a0_i}
			);
			vector<complex<double>> spinSusceptibility = susceptibilityCalculators[worker]->calculateSpinRPASusceptibility(
				kDual,
//				{c1_i, a0_i, c0_o, a1_o}
				{c0_o, a1_o, c1_i, a0_i}
			);
			for(
				unsigned int n = 0;
				n < numSummationEnergies;
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
/*			a1_i != orbitalIndices.at(3)
			|| c0_i != orbitalIndices.at(0)*/
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
/*				a0_o != orbitalIndices.at(2)
				|| c1_o != orbitalIndices.at(1)*/
				a0_o != orbitalIndices.at(0)
				|| c1_o != orbitalIndices.at(1)
				|| abs(amplitude_o) < 1e-10
			){
				continue;
			}

			vector<complex<double>> spinSusceptibility = susceptibilityCalculators[worker]->calculateSpinRPASusceptibility(
				kDual,
//				{c1_i, a0_i, c0_o, a1_o}
				{c0_o, a1_o, c1_i, a0_i}
			);
			for(
				unsigned int n = 0;
				n < numSummationEnergies;
				n++
			){
				selfEnergyVertex.at(n) += amplitude_i*amplitude_o*spinSusceptibility.at(n);
			}
		}
	}

	selfEnergyVertexTrees[worker].add(
		selfEnergyVertex,
		resultIndex
	);

	return selfEnergyVertex;
}

vector<complex<double>> SelfEnergyCalculator::calculateSelfEnergy(
	const vector<double> &k,
	const vector<int> &orbitalIndices
){
	TBTKAssert(
		isInitialized,
		"SelfEnergyCalculator::calculateSelfEnergy()",
		"SelfEnergyCalculator not yet initialized.",
		"Use SelfEnergyCalculator::init() to initialize the"
		<< " SelfEnergyCalculator."
	);
	TBTKAssert(
		orbitalIndices.size() == 2,
		"calculateSelfEnergy()",
		"Two orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	generateInteractionAmplitudes();

	const MomentumSpaceContext &momentumSpaceContext = susceptibilityCalculators[0]->getMomentumSpaceContext();
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
			orbitalIndices.at(1)
		}
	);

	SerializeableVector<complex<double>> result;
	if(selfEnergyTree.get(result, resultIndex))
		return result;

	//Initialize results
	for(unsigned int n = 0; n < selfEnergyEnergies.size(); n++)
		result.push_back(0);

	if(selfEnergyEnergies.size() == 1)
		selfEnergyMainLoop<true>(k, orbitalIndices, result);
	else
		selfEnergyMainLoop<false>(k, orbitalIndices, result);

	selfEnergyTree.add(
		result,
		resultIndex
	);

	return result;
}

vector<complex<double>> SelfEnergyCalculator::calculateSelfEnergySelfConsistently(
	unsigned int numMatsubaraFrequencies
){
	TBTKNotYetImplemented("SelfEnergyCalculator::calculateSelfEnergySelfConsistently");

	TBTKAssert(
		isInitialized,
		"SelfEnergyCalculator::calculateSelfEnergySelfConsistently()",
		"SelfEnergyCalculator not yet initialized.",
		"Use SelfEnergyCalculator::init() to initialize the"
		<< " SelfEnergyCalculator."
	);

	generateInteractionAmplitudes();

	const MomentumSpaceContext &momentumSpaceContext = susceptibilityCalculators[0]->getMomentumSpaceContext();
	const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();

	//Calculate offset
/*	Index kIndex = brillouinZone.getMinorCellIndex(
		k,
		numMeshPoints
	);

	Index resultIndex = Index(
		kIndex,
		{
			orbitalIndices.at(0),
			orbitalIndices.at(1)
		}
	);*/

	SerializeableVector<complex<double>> result;
/*	if(selfEnergyTree.get(result, resultIndex))
		return result;

	//Initialize results
	for(unsigned int n = 0; n < selfEnergyEnergies.size(); n++)
		result.push_back(0);

	if(selfEnergyEnergies.size() == 1)
		selfEnergyMainLoop<true>(k, orbitalIndices, result);
	else
		selfEnergyMainLoop<false>(k, orbitalIndices, result);

	selfEnergyTree.add(
		result,
		resultIndex
	);*/

	return result;
}

template<bool singleSelfEnergyEnergy>
void SelfEnergyCalculator::selfEnergyMainLoop(
	const vector<double> &k,
	const vector<int> &orbitalIndices,
	vector<complex<double>> &result
){
	generateKMinusQLookupTable();

	const MomentumSpaceContext &momentumSpaceContext = susceptibilityCalculators[0]->getMomentumSpaceContext();
	const Model &model = momentumSpaceContext.getModel();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	//Get kIndex
	Index kIndex = momentumSpaceContext.getKIndex(k);

	//Get linear index corresponding to kIndex.
	int kLinearIndex = model.getHoppingAmplitudeSet()->getFirstIndexInBlock(
		kIndex
	);

	vector<vector<complex<double>>> results;
	results.reserve(susceptibilityCalculators.size());
	for(unsigned int n = 0; n < susceptibilityCalculators.size(); n++){
		results.push_back(vector<complex<double>>());
		results[n].reserve(results.size());
		for(unsigned int c = 0; c < result.size(); c++)
			results[n].push_back(0);
	}

	//Main loop
	#pragma omp parallel for default(none) shared(mesh, kLinearIndex, k, numOrbitals, orbitalIndices, momentumSpaceContext, model, results)
	for(unsigned int worker = 0; worker < susceptibilityCalculators.size(); worker++){
		unsigned int blockSize = mesh.size()/susceptibilityCalculators.size();
		unsigned int begin = worker*blockSize;
		unsigned int end = (worker+1)*blockSize;
		if(worker == susceptibilityCalculators.size()-1)
			end = mesh.size();

		for(unsigned int n = begin; n < end; n++){
//		for(unsigned int n = 0; n < mesh.size(); n++){
			//Get linear index corresponding to k-q
			int kMinusQLinearIndex = getKMinusQLinearIndex<true>(
				n,
				k,
				kLinearIndex
			);
			int kMinusQMeshPoint = kMinusQLinearIndex/numOrbitals;

			for(
				unsigned int propagatorStart = 0;
				propagatorStart < numOrbitals;
				propagatorStart++
			){
				for(
					unsigned int propagatorEnd = 0;
					propagatorEnd < numOrbitals;
					propagatorEnd++
				){
					vector<complex<double>> selfEnergyVertex = calculateSelfEnergyVertex(
						mesh.at(n),
						{
							(int)propagatorEnd,
							orbitalIndices[0],
							(int)propagatorStart,
							orbitalIndices[1]
						},
						worker
					);

					for(
						unsigned int state = 0;
						state < numOrbitals;
						state++
					){
						double e = momentumSpaceContext.getEnergy(
							kMinusQLinearIndex + state
						);
						complex<double> a0 = momentumSpaceContext.getAmplitude(
							kMinusQMeshPoint,
							state,
							propagatorEnd
						);
						complex<double> a1 = momentumSpaceContext.getAmplitude(
							kMinusQMeshPoint,
							state,
							propagatorStart
						);

						complex<double> greensFunctionNumerator = a0*conj(a1);
						double relativeStateEnergy = e - model.getChemicalPotential();

						for(
							unsigned int e0 = 0;
							e0 < numSummationEnergies;
							e0++
						){
							complex<double> numerator = selfEnergyVertex[e0]*greensFunctionNumerator;
							complex<double> E = summationEnergies[e0] - relativeStateEnergy;

							if(singleSelfEnergyEnergy){
								results[worker][0] += numerator/(
									selfEnergyEnergies[0] + E
								);
							}
							else{
								for(
									unsigned int e1 = 0;
									e1 < selfEnergyEnergies.size();
									e1++
								){
									results[worker][e1] += numerator/(
										selfEnergyEnergies[e1] + E
									);
								}
							}
						}
					}
				}
			}
		}
	}

	for(unsigned int n = 0; n < susceptibilityCalculators.size(); n++)
		for(unsigned int c = 0; c < result.size(); c++)
			result[c] += results[n][c];

	//Calculate kT
	double temperature = UnitHandler::convertTemperatureNtB(
		susceptibilityCalculators[0]->getMomentumSpaceContext(
		).getModel().getTemperature()
	);
	double kT = UnitHandler::getK_BB()*temperature;

	for(unsigned int n = 0; n < selfEnergyEnergies.size(); n++)
		result.at(n) *= kT/mesh.size();
}

}	//End of namesapce TBTK
