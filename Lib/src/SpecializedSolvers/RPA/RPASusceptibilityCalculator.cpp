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

/** @file SusceptibilityCalculator.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Functions.h"
#include "TBTK/RPA/MatsubaraSusceptibilityCalculator.h"
#include "TBTK/RPA/RPASusceptibilityCalculator.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

//const complex<double> i(0, 1);

namespace TBTK{

RPASusceptibilityCalculator::RPASusceptibilityCalculator(
	const RPA::MomentumSpaceContext &momentumSpaceContext,
	SusceptibilityCalculator::Algorithm algorithm
){
	switch(algorithm){
	case SusceptibilityCalculator::Algorithm::Lindhard:
		susceptibilityCalculator = new LindhardSusceptibilityCalculator(
			momentumSpaceContext
		);
	case SusceptibilityCalculator::Algorithm::Matsubara:
		susceptibilityCalculator = new MatsubaraSusceptibilityCalculator(
			momentumSpaceContext
		);
	default:
		TBTKExit(
			"RPASusceptibilityCalculator::RPASusceptibilityCalculator()",
			"Unknown algorithm.",
			"This should never happen, contact the developer."
		);
	}

	U = 0.;
	Up = 0.;
	J = 0.;
	Jp = 0.;

	susceptibilityCalculator->setEnergyType(
		SusceptibilityCalculator::EnergyType::Complex
	);

	interactionAmplitudesAreGenerated = false;
}

RPASusceptibilityCalculator::RPASusceptibilityCalculator(
	SusceptibilityCalculator &susceptibilityCalculator
//	LindhardSusceptibilityCalculator &lindhardSusceptibilityCalculator
){
	this->susceptibilityCalculator = susceptibilityCalculator.createSlave();

	U = 0.;
	Up = 0.;
	J = 0.;
	Jp = 0.;

	interactionAmplitudesAreGenerated = false;
}

RPASusceptibilityCalculator::~RPASusceptibilityCalculator(){
	if(susceptibilityCalculator != nullptr)
		delete susceptibilityCalculator;
}

RPASusceptibilityCalculator* RPASusceptibilityCalculator::createSlave(){
	return new RPASusceptibilityCalculator(
		*susceptibilityCalculator
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

inline void RPASusceptibilityCalculator::invertMatrix(
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

void RPASusceptibilityCalculator::multiplyMatrices(
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

vector<vector<vector<complex<double>>>> RPASusceptibilityCalculator::rpaSusceptibilityMainAlgorithm(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices,
	const vector<InteractionAmplitude> &interactionAmplitudes
){
	TBTKAssert(
		orbitalIndices.size() == 4,
		"SusceptibilityCalculator::rpaSusceptibilityMainAlgorithm()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	const RPA::MomentumSpaceContext &momentumSpaceContext
		= susceptibilityCalculator->getMomentumSpaceContext();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();
	unsigned int matrixDimension = numOrbitals*numOrbitals;
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

				vector<complex<double>> susceptibility
					= susceptibilityCalculator->calculateSusceptibility(
						kDual,
						{c1, a0, (int)d, (int)c}
					);
				for(
					unsigned int i = 0;
					i < energies.size();
					i++
				){
					denominators.at(i)[
						matrixDimension*col + row
					] += amplitude*susceptibility.at(i);
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
			vector<complex<double>> susceptibility
				= susceptibilityCalculator->calculateSusceptibility(
					kDual,
					{
						orbitalIndices.at(0),
						orbitalIndices.at(1),
						(int)d,
						(int)c
					}
				);
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
						]*susceptibility.at(i);
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

vector<complex<double>> RPASusceptibilityCalculator::calculateRPASusceptibility(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
	TBTKAssert(
		orbitalIndices.size() == 4,
		"getSusceptibility()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	//Get kIndex and resultIndex
	Index resultIndex = getSusceptibilityResultIndex(
		kDual,
		orbitalIndices
	);

	//Try to return cashed result
	SerializableVector<complex<double>> result;
	if(rpaSusceptibilityTree.get(result, resultIndex))
		return result;

	//Calculate RPA-susceptibility
	vector<vector<vector<complex<double>>>> rpaSusceptibility = rpaSusceptibilityMainAlgorithm(
		kDual,
		orbitalIndices,
		interactionAmplitudes
	);

	//Cache result
	const RPA::MomentumSpaceContext &momentumSpaceContext
		= susceptibilityCalculator->getMomentumSpaceContext();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();
	for(unsigned int orbital0 = 0; orbital0 < numOrbitals; orbital0++){
		for(unsigned int orbital1 = 0; orbital1 < numOrbitals; orbital1++){
			Index resultIndex = getSusceptibilityResultIndex(
				kDual,
				{
					orbitalIndices[0],
					orbitalIndices[1],
					(int)orbital0,
					(int)orbital1
				}
			);
			rpaSusceptibilityTree.add(
				rpaSusceptibility[orbital0][orbital1],
				resultIndex
			);
		}
	}
	result = rpaSusceptibility[orbitalIndices[2]][orbitalIndices[3]];

	return result;
}

void RPASusceptibilityCalculator::generateInteractionAmplitudes(){
	if(interactionAmplitudesAreGenerated)
		return;
//	Streams::out << "Generating interaction amplitudes.\n";

	interactionAmplitudesCharge.clear();
	interactionAmplitudesSpin.clear();

	const RPA::MomentumSpaceContext &momentumSpaceContext
		= susceptibilityCalculator->getMomentumSpaceContext();
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

vector<complex<double>> RPASusceptibilityCalculator::calculateChargeRPASusceptibility(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
	TBTKAssert(
		orbitalIndices.size() == 4,
		"getSusceptibility()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	//Get kIndex and resultIndex
	Index resultIndex = getSusceptibilityResultIndex(
		kDual,
		orbitalIndices
	);

	//Try to return cashed result
	SerializableVector<complex<double>> result;
	if(rpaChargeSusceptibilityTree.get(result, resultIndex))
		return result;

	const RPA::MomentumSpaceContext &momentumSpaceContext
		= susceptibilityCalculator->getMomentumSpaceContext();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	//Setup InteractionAmplitude
	generateInteractionAmplitudes();

	//Calculate the RPA charge-susceptibility
	vector<vector<vector<complex<double>>>> rpaSusceptibility = rpaSusceptibilityMainAlgorithm(
		kDual,
		orbitalIndices,
		interactionAmplitudesCharge
	);

	//Cache result
	for(unsigned int orbital0 = 0; orbital0 < numOrbitals; orbital0++){
		for(unsigned int orbital1 = 0; orbital1 < numOrbitals; orbital1++){
			Index resultIndex = getSusceptibilityResultIndex(
				kDual,
				{
					orbitalIndices[0],
					orbitalIndices[1],
					(int)orbital0,
					(int)orbital1
				}
			);
			rpaChargeSusceptibilityTree.add(
				rpaSusceptibility[orbital0][orbital1],
				resultIndex
			);

			//<!!!Needs proper checking!!!>
			const RPA::MomentumSpaceContext &momentumSpaceContext
				= susceptibilityCalculator->getMomentumSpaceContext();
			const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();
			const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
//			const vector<double> &k = kDual.getContinuousIndex();
			const vector<double> &k = kDual;

			const vector<complex<double>> &result = rpaSusceptibility[orbital0][orbital1];

			//Use symmetries to extend result to other entries.
			if(
				getEnergyType() == EnergyType::Imaginary
				&& getEnergiesAreInversionSymmetric()
			){
				vector<complex<double>> conjugatedResult;
				for(unsigned int n = 0; n < result.size(); n++){
					conjugatedResult.push_back(
						conj(result.at(n))
					);
				}

				vector<double> kMinus;
				for(unsigned int n = 0; n < k.size(); n++)
					kMinus.push_back(-k.at(n));
				Index kMinusIndex = brillouinZone.getMinorCellIndex(
					kMinus,
					numMeshPoints
				);

				rpaChargeSusceptibilityTree.add(
					conjugatedResult,
					Index(
						kMinusIndex,
						{
							orbitalIndices.at(1),
							orbitalIndices.at(0),
							(int)orbital1,
							(int)orbital0
						}
					)
				);
			}
			//</!!!Needs proper checking!!!>
		}
	}
	result = rpaSusceptibility[orbitalIndices[2]][orbitalIndices[3]];

	return result;
}

vector<complex<double>> RPASusceptibilityCalculator::calculateSpinRPASusceptibility(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
	TBTKAssert(
		orbitalIndices.size() == 4,
		"getSusceptibility()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	//Get kIndex and resultIndex
	Index resultIndex = getSusceptibilityResultIndex(
		kDual,
		orbitalIndices
	);

	//Try to return cached result
	SerializableVector<complex<double>> result;
	if(rpaSpinSusceptibilityTree.get(result, resultIndex))
		return result;

	const RPA::MomentumSpaceContext &momentumSpaceContext
		= susceptibilityCalculator->getMomentumSpaceContext();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	//Setup InteractionAmplitude
	generateInteractionAmplitudes();

	//Calculate RPA spin-susceptibility
	vector<vector<vector<complex<double>>>> rpaSusceptibility = rpaSusceptibilityMainAlgorithm(
		kDual,
		orbitalIndices,
		interactionAmplitudesSpin
	);

	//Cache result
	for(unsigned int orbital0 = 0; orbital0 < numOrbitals; orbital0++){
		for(unsigned int orbital1 = 0; orbital1 < numOrbitals; orbital1++){
			Index resultIndex = getSusceptibilityResultIndex(
				kDual,
				{
					orbitalIndices[0],
					orbitalIndices[1],
					(int)orbital0,
					(int)orbital1
				}
			);
			rpaSpinSusceptibilityTree.add(
				rpaSusceptibility[orbital0][orbital1],
				resultIndex
			);

			//<!!!Needs proper checking!!!>
			const RPA::MomentumSpaceContext &momentumSpaceContext
				= susceptibilityCalculator->getMomentumSpaceContext();
			const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();
			const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
//			const vector<double> &k = kDual.getContinuousIndex();
			const vector<double> &k = kDual;

			const vector<complex<double>> &result = rpaSusceptibility[orbital0][orbital1];

			//Use symmetries to extend result to other entries.
			if(
				getEnergyType() == EnergyType::Imaginary
				&& getEnergiesAreInversionSymmetric()
			){
				vector<complex<double>> conjugatedResult;
				for(unsigned int n = 0; n < result.size(); n++){
					conjugatedResult.push_back(
						conj(result.at(n))
					);
				}

				vector<double> kMinus;
				for(unsigned int n = 0; n < k.size(); n++)
					kMinus.push_back(-k.at(n));
				Index kMinusIndex = brillouinZone.getMinorCellIndex(
					kMinus,
					numMeshPoints
				);

				rpaSpinSusceptibilityTree.add(
					conjugatedResult,
					Index(
						kMinusIndex,
						{
							orbitalIndices.at(1),
							orbitalIndices.at(0),
							(int)orbital1,
							(int)orbital0
						}
					)
				);
			}
			//</!!!Needs proper checking!!!>
		}
	}
	result = rpaSusceptibility[orbitalIndices[2]][orbitalIndices[3]];

	return result;
}

}	//End of namesapce TBTK
