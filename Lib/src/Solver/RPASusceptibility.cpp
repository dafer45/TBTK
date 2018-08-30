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

/** @file RPASusceptibility.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Functions.h"
#include "TBTK/Solver/RPASusceptibility.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

namespace TBTK{
namespace Solver{

RPASusceptibility::RPASusceptibility(
	const MomentumSpaceContext &momentumSpaceContext,
	const Property::Susceptibility &bareSusceptibility
) :
	Susceptibility(Algorithm::RPA, momentumSpaceContext),
	Communicator(true),
	bareSusceptibility(bareSusceptibility)
{
	U = 0.;
	Up = 0.;
	J = 0.;
	Jp = 0.;

	interactionAmplitudesAreGenerated = false;
}

RPASusceptibility* RPASusceptibility::createSlave(){
	TBTKExit(
		"Solver::RPASusceptibility::createSlave()",
		"This function is not supported by this solver.",
		""
	);
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

inline void RPASusceptibility::invertMatrix(
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

void RPASusceptibility::multiplyMatrices(
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

void printMatrix(complex<double> *matrix, unsigned int dimension){
	for(unsigned int r = 0; r < dimension; r++){
		for(unsigned int c = 0; c < dimension; c++){
			Streams::out << setw(20) << matrix[dimension*c + r];
		}
		Streams::out << "\n";
	}
	Streams::out << "\n";
}

vector<vector<vector<complex<double>>>> RPASusceptibility::rpaSusceptibilityMainAlgorithm(
	const Index &index,
	const vector<InteractionAmplitude> &interactionAmplitudes
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 5,
		"SusceptibilityCalculator::rpaSusceptibilityMainAlgorithm()",
		"The Index must be a compound Index with 5 component Indices,"
		<< " but '" << components.size() << "' components suplied.",
		""
	);
	Index kIndex = components[0];
	Index intraBlockIndices[4] = {
		components[1],
		components[2],
		components[3],
		components[4],
	};

	const MomentumSpaceContext &momentumSpaceContext
		= getMomentumSpaceContext();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();
	unsigned int matrixDimension = numOrbitals*numOrbitals;

	//Setup energies.
	vector<complex<double>> energies;
	switch(bareSusceptibility.getEnergyType()){
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::Real:
		for(
			unsigned int n = 0;
			n < bareSusceptibility.getResolution();
			n++
		){
			energies.push_back(bareSusceptibility.getEnergy(n));
		}
		break;
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::BosonicMatsubara:
		for(
			unsigned int n = 0;
			n < bareSusceptibility.getNumMatsubaraEnergies();
			n++
		){
			energies.push_back(
				bareSusceptibility.getMatsubaraEnergy(n)
			);
		}
		break;
	default:
		TBTKExit(
			"Solver::RPASusceptibility::calculateSusceptibilityMainAlgorithm()",
			"Only the energy types"
			" Property::EnergyResolvedProperty::EnergyType::Real and"
			<< " Property::EnergyResolvedProperty::EnergyType::BosonicMatsubara"
			<< " are supported, but the bare susceptibility has a"
			<< " different energy type.",
			"This should never happen, contact the developer."
		);
	}

	//Denominator in the expression chi_RPA = chi_0/(1 - U\chi_0)
	vector<complex<double>*> denominators;

	//Initialize denominator matrices to unit matrices
	for(
		unsigned int e = 0;
		e < energies.size();
		e++
	){
		//Create denominator matrix
		denominators.push_back(
			new complex<double>[matrixDimension*matrixDimension]
		);
		//Initialize denominator matrices to unit matrices
		for(
			unsigned int c = 0;
			c < matrixDimension*matrixDimension;
			c++
		){
			denominators.at(e)[c] = 0.;
		}
		for(unsigned int c = 0; c < matrixDimension; c++)
			denominators.at(e)[c*matrixDimension + c] = 1.;
	}

	//Calculate denominator = (1 + U\chi_0)
	for(unsigned int n = 0; n < interactionAmplitudes.size(); n++){
		const InteractionAmplitude &interactionAmplitude = interactionAmplitudes.at(n);

		int c0 = interactionAmplitude.getCreationOperatorIndex(0).at(0);
		int c1 = interactionAmplitude.getCreationOperatorIndex(1).at(0);
		int a0 = interactionAmplitude.getAnnihilationOperatorIndex(0).at(0);
		int a1 = interactionAmplitude.getAnnihilationOperatorIndex(1).at(0);
		complex<double> amplitude = interactionAmplitude.getAmplitude();

		if(abs(amplitude) < 1e-10)
			continue;

		int row = numOrbitals*c0 + a1;
		for(unsigned int c = 0; c < numOrbitals; c++){
			for(unsigned int d = 0; d < numOrbitals; d++){
				int col = numOrbitals*c + d;

/*				vector<complex<double>> susceptibility
					= susceptibilityCalculator->calculateSusceptibility(
						kDual,
						{c1, a0, (int)d, (int)c}
					);*/
				const vector<complex<double>> &susceptibility
					= bareSusceptibility.getData();
				unsigned int offset
					= bareSusceptibility.getOffset({
						kIndex,
						{c1},
						{a0},
						{(int)d},
						{(int)c}
					});
				for(
					unsigned int i = 0;
					i < energies.size();
					i++
				){
					denominators.at(i)[
						matrixDimension*col + row
					] += amplitude*susceptibility.at(offset + i);
				}
			}
		}
	}

	//Calculate (1 + U\chi_0)^{-1}
	for(unsigned int n = 0; n < energies.size(); n++)
		invertMatrix(denominators.at(n), matrixDimension);

	//Initialize \chi_RPA
	vector<vector<vector<complex<double>>>> rpaSusceptibility;
	for(unsigned int orbital2 = 0; orbital2 < numOrbitals; orbital2++){
		rpaSusceptibility.push_back(vector<vector<complex<double>>>());
		for(unsigned int orbital3 = 0; orbital3 < numOrbitals; orbital3++){
			rpaSusceptibility[orbital2].push_back(vector<complex<double>>());
			for(
				unsigned int e = 0;
				e < energies.size();
				e++
			){
				rpaSusceptibility[orbital2][orbital3].push_back(0.);
			}
		}
	}

	//Calculate \chi_RPA = \chi_0/(1 + U\chi_0)
	for(unsigned int c = 0; c < numOrbitals; c++){
		for(unsigned int d = 0; d < numOrbitals; d++){
/*			vector<complex<double>> susceptibility
				= susceptibilityCalculator->calculateSusceptibility(
					kDual,
					{
						orbitalIndices.at(0),
						orbitalIndices.at(1),
						(int)d,
						(int)c
					}
				);*/
			const vector<complex<double>> &susceptibility
				= bareSusceptibility.getData();
			unsigned int offset = bareSusceptibility.getOffset({
				kIndex,
				intraBlockIndices[0],
				intraBlockIndices[1],
				{(int)d},
				{(int)c}
			});
			for(unsigned int orbital2 = 0; orbital2 < numOrbitals; orbital2++){
				for(unsigned int orbital3 = 0; orbital3 < numOrbitals; orbital3++){
					for(
						unsigned int i = 0;
						i < energies.size();
						i++
					){
						rpaSusceptibility[orbital2][orbital3].at(i) += denominators.at(i)[
							matrixDimension*(
								numOrbitals*orbital2
								+ orbital3
							) + numOrbitals*c + d
						]*susceptibility.at(offset + i);
					}
				}
			}
		}
	}

	//Free memory allocated for denominators
	for(unsigned int n = 0; n < energies.size(); n++)
		delete [] denominators.at(n);

	return rpaSusceptibility;
}

IndexedDataTree<vector<complex<double>>> RPASusceptibility::calculateRPASusceptibility(
	const Index &index
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 5,
		"SusceptibilityCalculator::calculateRPASusceptibility()",
		"The Index must be a compound Index with 5 component Indices,"
		<< " but '" << components.size() << "' components suplied.",
		""
	);
	Index kIndex = components[0];
	Index intraBlockIndices[4] = {
		components[1],
		components[2],
		components[3],
		components[4],
	};

	//Calculate RPA-susceptibility
/*	vector<vector<vector<complex<double>>>> rpaSusceptibility = rpaSusceptibilityMainAlgorithm(
		kDual,
		orbitalIndices,
		interactionAmplitudes
	);*/

	//TODO
	//The way intraBlockIndices[n] are used assumes that they have a single
	//subindex, which limits generality.
	vector<vector<vector<complex<double>>>> result = rpaSusceptibilityMainAlgorithm(
		index,
		interactionAmplitudes
	);
	IndexedDataTree<vector<complex<double>>> indexedDataTree;
	for(unsigned int n = 0; n < result.size(); n++){
		for(unsigned int c = 0; c < result[n].size(); c++){
			indexedDataTree.add(
				result[n][c],
				{
					kIndex,
					intraBlockIndices[0],
					intraBlockIndices[1],
					{(int)n},
					{(int)c}
				}
			);
		}
	}

	return indexedDataTree;
}

void RPASusceptibility::generateInteractionAmplitudes(){
	if(interactionAmplitudesAreGenerated)
		return;

	interactionAmplitudesCharge.clear();
	interactionAmplitudesSpin.clear();

	const MomentumSpaceContext &momentumSpaceContext
		= getMomentumSpaceContext();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	//Generate charge-interaction amplitudes.
	for(int a = 0; a < (int)numOrbitals; a++){
		interactionAmplitudesCharge.push_back(
			InteractionAmplitude(
				U,
				{{a},	{a}},
				{{a},	{a}}
			)
		);

		for(int b = 0; b < (int)numOrbitals; b++){
			if(a == b)
				continue;

			interactionAmplitudesCharge.push_back(
				InteractionAmplitude(
					2.*Up - J,
					{{a},	{b}},
					{{b},	{a}}
				)
			);
			interactionAmplitudesCharge.push_back(
				InteractionAmplitude(
					-Up + 2.*J,
					{{a},	{b}},
					{{a},	{b}}
				)
			);
			interactionAmplitudesCharge.push_back(
				InteractionAmplitude(
					Jp,
					{{a},	{a}},
					{{b},	{b}}
				)
			);
		}
	}

	//Generate spin-interaction amplitudes.
	for(int a = 0; a < (int)numOrbitals; a++){
		interactionAmplitudesSpin.push_back(
			InteractionAmplitude(
				-U,
				{{a},	{a}},
				{{a},	{a}}
			)
		);

		for(int b = 0; b < (int)numOrbitals; b++){
			if(a == b)
				continue;

			interactionAmplitudesSpin.push_back(
				InteractionAmplitude(
					-J,
					{{a},	{b}},
					{{b},	{a}}
				)
			);
			interactionAmplitudesSpin.push_back(
				InteractionAmplitude(
					-Up,
					{{a},	{b}},
					{{a},	{b}}
				)
			);
			interactionAmplitudesSpin.push_back(
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

IndexedDataTree<vector<complex<double>>> RPASusceptibility::calculateChargeRPASusceptibility(
	const Index &index
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 5,
		"SusceptibilityCalculator::calculateChargeRPASusceptibility()",
		"The Index must be a compound Index with 5 component Indices,"
		<< " but '" << components.size() << "' components suplied.",
		""
	);
	Index kIndex = components[0];
	Index intraBlockIndices[4] = {
		components[1],
		components[2],
		components[3],
		components[4],
	};

	//Setup InteractionAmplitude
	generateInteractionAmplitudes();

	//Calculate the RPA charge-susceptibility
/*	vector<vector<vector<complex<double>>>> rpaSusceptibility = rpaSusceptibilityMainAlgorithm(
		kDual,
		orbitalIndices,
		interactionAmplitudesCharge
	);*/

/*	return rpaSusceptibilityMainAlgorithm(
		index,
		interactionAmplitudesCharge
	)[intraBlockIndices[2][0]][intraBlockIndices[3][0]];*/

	//TODO
	//The way intraBlockIndices[n] are used assumes that they have a single
	//subindex, which limits generality.
	vector<vector<vector<complex<double>>>> result = rpaSusceptibilityMainAlgorithm(
		index,
		interactionAmplitudesCharge
	);
	IndexedDataTree<vector<complex<double>>> indexedDataTree;
	for(unsigned int n = 0; n < result.size(); n++){
		for(unsigned int c = 0; c < result[n].size(); c++){
			indexedDataTree.add(
				result[n][c],
				{
					kIndex,
					intraBlockIndices[0],
					intraBlockIndices[1],
					{(int)n},
					{(int)c}
				}
			);
		}
	}

	return indexedDataTree;
}

IndexedDataTree<vector<complex<double>>> RPASusceptibility::calculateSpinRPASusceptibility(
	const Index &index
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 5,
		"SusceptibilityCalculator::calculateSpinRPASusceptibility()",
		"The Index must be a compound Index with 5 component Indices,"
		<< " but '" << components.size() << "' components suplied.",
		""
	);
	Index kIndex = components[0];
	Index intraBlockIndices[4] = {
		components[1],
		components[2],
		components[3],
		components[4],
	};

	//Setup InteractionAmplitude
	generateInteractionAmplitudes();

	//Calculate RPA spin-susceptibility
/*	vector<vector<vector<complex<double>>>> rpaSusceptibility = rpaSusceptibilityMainAlgorithm(
		kDual,
		orbitalIndices,
		interactionAmplitudesSpin
	);*/

/*	return rpaSusceptibilityMainAlgorithm(
		index,
		interactionAmplitudesSpin
	)[intraBlockIndices[2][0]][intraBlockIndices[3][0]];*/

	//TODO
	//The way intraBlockIndices[n] are used assumes that they have a single
	//subindex, which limits generality.
	vector<vector<vector<complex<double>>>> result = rpaSusceptibilityMainAlgorithm(
		index,
		interactionAmplitudesSpin
	);
	IndexedDataTree<vector<complex<double>>> indexedDataTree;
	for(unsigned int n = 0; n < result.size(); n++){
		for(unsigned int c = 0; c < result[n].size(); c++){
			indexedDataTree.add(
				result[n][c],
				{
					kIndex,
					intraBlockIndices[0],
					intraBlockIndices[1],
					{(int)n},
					{(int)c}
				}
			);
		}
	}

	return indexedDataTree;
}

}	//End of namespace Solver
}	//End of namesapce TBTK
