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

/** @file ZFactorCalculator.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Functions.h"
#include "TBTK/RPA/ZFactorCalculator.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

const complex<double> i(0, 1);

namespace TBTK{

ZFactorCalculator::ZFactorCalculator(
	const RPA::MomentumSpaceContext &momentumSpaceContext,
	unsigned int numWorkers
) : selfEnergyCalculator(momentumSpaceContext, numWorkers){
	isInitialized = false;

	numSummationEnergies = 0;

	U = 0.;
	Up = 0.;
	J = 0.;
	Jp = 0.;

	const Model &model = momentumSpaceContext.getModel();

	//Calculate kT
	double temperature
		= UnitHandler::convertNaturalToBase<Quantity::Temperature>(
			model.getTemperature()
		);
	double kT = UnitHandler::getConstantInBaseUnits("k_B")*temperature;

	//Setup self-energy energies
/*	selfEnergyCalculator.setEnergyType(
		SelfEnergyCalculator::EnergyType::Imaginary
	);*/
	selfEnergyCalculator.setSelfEnergyEnergies({i*M_PI*kT});
}

ZFactorCalculator::~ZFactorCalculator(){
}

void ZFactorCalculator::init(){
	TBTKAssert(
		numSummationEnergies%2 == 1,
		"ZFactorCalculator::init()",
		"The number of summation energies must be an odd number.",
		"Use ZFactor::setNumSummationEnergies() to set the number of"
		<< " summation energies."
	);


	//Setup SelfEnergyCalculator.
	selfEnergyCalculator.setNumSummationEnergies(numSummationEnergies);
	selfEnergyCalculator.init();

	isInitialized = true;
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

void ZFactorCalculator::invertMatrix(
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

void ZFactorCalculator::multiplyMatrices(
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

void ZFactorCalculator::printMatrix(complex<double> *matrix, unsigned int dimension){
	for(unsigned int r = 0; r < dimension; r++){
		for(unsigned int c = 0; c < dimension; c++){
			Streams::out << setw(20) << matrix[dimension*c + r];
		}
		Streams::out << "\n";
	}
	Streams::out << "\n";
}

vector<complex<double>> ZFactorCalculator::calculateZFactor(
	const vector<double> &k
){
	const RPA::MomentumSpaceContext &momentumSpaceContext = selfEnergyCalculator.getMomentumSpaceContext();
	const Model &model = momentumSpaceContext.getModel();
	const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();
	const PropertyExtractor::BlockDiagonalizer &propertyExtractor
		= momentumSpaceContext.getPropertyExtractorBlockDiagonalizer();

	//Calculate kT
	double temperature
		= UnitHandler::convertNaturalToBase<Quantity::Temperature>(
			model.getTemperature()
		);
	double kT = UnitHandler::getConstantInBaseUnits("k_B")*temperature;

	Index kIndex = brillouinZone.getMinorCellIndex(
		k,
		numMeshPoints
	);

	complex<double> *zFactor = new complex<double>[
		numOrbitals*numOrbitals
	];
	for(unsigned int orbital0 = 0; orbital0 < numOrbitals; orbital0++){
		for(
			unsigned int orbital1 = 0;
			orbital1 < numOrbitals;
			orbital1++
		){
/*			zFactor[numOrbitals*orbital1 + orbital0] = -imag(
				selfEnergyCalculator.calculateSelfEnergy(
					k,
					{(int)orbital0, (int)orbital1}
				).at(0)
			)/(M_PI*kT);*/
			zFactor[numOrbitals*orbital1 + orbital0]
				= -selfEnergyCalculator.calculateSelfEnergy(
					k,
					{(int)orbital0, (int)orbital1}
				).at(0)/(M_PI*kT);
		}
	}

/*	for(unsigned int n = 0; n < numOrbitals; n++)
		zFactor[numOrbitals*n + n] += 1.;*/

	complex<double> *eigenVectors = new complex<double>[
		numOrbitals*numOrbitals
	];
	complex<double> *eigenVectorsHermitianConjugate = new complex<double>[
		numOrbitals*numOrbitals
	];
	for(unsigned int row = 0; row < numOrbitals; row++){
		for(unsigned int col = 0; col < numOrbitals; col++){
			complex<double> amplitude = propertyExtractor.getAmplitude(
				kIndex,
				row,
				{(int)col}
			);

			eigenVectors[numOrbitals*col + row] = amplitude;
			eigenVectorsHermitianConjugate[
				numOrbitals*row + col
			] = conj(amplitude);
		}
	}

	complex<double> *zFactorWorkspace = new complex<double>[
		numOrbitals*numOrbitals
	];
	multiplyMatrices(eigenVectors, zFactor, zFactorWorkspace, numOrbitals);
	multiplyMatrices(
		zFactorWorkspace,
		eigenVectorsHermitianConjugate,
		zFactor,
		numOrbitals
	);

	for(unsigned int n = 0; n < numOrbitals*numOrbitals; n++)
		zFactor[n] = imag(zFactor[n]);

	double offDiagonals = 0;
	double diagonals = 0;
	for(unsigned int r = 0; r < numOrbitals; r++){
		for(unsigned int c = 0; c < numOrbitals; c++){
			if(r == c)
				diagonals += abs(zFactor[numOrbitals*r + c]);
			else
				offDiagonals += abs(zFactor[numOrbitals*r + c]);
		}
	}
	Streams::out << "Self-energy off-diagonals over diagonals:\t" << offDiagonals/diagonals << "\n";

	for(unsigned int n = 0; n < numOrbitals; n++)
		zFactor[numOrbitals*n + n] += 1.;

	invertMatrix(zFactor, numOrbitals);

	vector<complex<double>> zFactors;
	for(unsigned int n = 0; n < numOrbitals; n++)
		zFactors.push_back(zFactor[numOrbitals*n + n]);

	delete [] zFactor;
	delete [] zFactorWorkspace;
	delete [] eigenVectors;
	delete [] eigenVectorsHermitianConjugate;

	return zFactors;
}

/*vector<complex<double>> ZFactorCalculator::calculateZFactor2(
	const vector<double> &k
){
	TBTKAssert(
		isInitialized,
		"ZFactorCalculator::calculateSusceptibility2()",
		"ZFactorCalculator not yet initialized.",
		"Use ZFactorCalculator::init() to initialize the"
		<< " ZFactorCalculator."
	);

	//<!!!Temporarily defined here!!!>
	complex<double> eta = 0.06;
	//</!!!Temporarily defined here>

	const MomentumSpaceContext &momentumSpaceContext = selfEnergyCalculator.getMomentumSpaceContext();
	const Model &model = momentumSpaceContext.getModel();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();
	const BPropertyExtractor &propertyExtractor = momentumSpaceContext.getPropertyExtractor();
	Index kIndex = momentumSpaceContext.getKIndex(k);

	//Basis size
//	int basisSize = model.getBasisSize();

	complex<double> *zFactor = new complex<double>[numOrbitals*numOrbitals];
	for(unsigned int n = 0; n < numOrbitals*numOrbitals; n++)
		zFactor[n] = 0.;

//	Timer::tick("Main loop");
//	int numMatsubaraFrequencies = 51;
	double t = UnitHandler::convertTemperatureNtB(model.getTemperature());
	double kT = UnitHandler::getK_BB()*t;

	//Initialize summation energies
	vector<complex<double>> summationEnergies;
	for(
		int n = -(int)numSummationEnergies/2;
		n <= (int)numSummationEnergies/2;
		n++
	){
		summationEnergies.push_back(i*M_PI*2.*(double)(n)*kT);
	}

	for(unsigned int n = 0; n < mesh.size(); n++){
		Index kMinusQIndex = brillouinZone.getMinorCellIndex(
			{mesh[n][0] - k[0], mesh[n][1] - k[1]},
			numMeshPoints
		);
		int kMinusQLinearIndex =  model.getHoppingAmplitudeSet()->getFirstIndexInBlock(
			kMinusQIndex
		);

		for(
			unsigned int incommingIndex = 0;
			incommingIndex < numOrbitals;
			incommingIndex++
		){
			for(
				unsigned int outgoingIndex = 0;
				outgoingIndex < numOrbitals;
				outgoingIndex++
			){
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
						vector<complex<double>> selfEnergyVertex = selfEnergyCalculator.calculateSelfEnergyVertex(
							mesh.at(n),
							{
								(int)propagatorStart,
								(int)incommingIndex,
								(int)propagatorEnd,
								(int)outgoingIndex
							},
							0
						);

						for(
							unsigned int state = 0;
							state < numOrbitals;
							state++
						){
//							double e = energies[(kMinusQLinearIndex + basisSize + (int)state)%basisSize];
//							complex<double> a0 = amplitudes[numOrbitals*((kMinusQLinearIndex + basisSize + state)%basisSize) + propagatorIndex0];
//							complex<double> a1 = amplitudes[numOrbitals*((kMinusQLinearIndex + basisSize + state)%basisSize) + propagatorIndex1];
							double e = momentumSpaceContext.getEnergy(
								kMinusQLinearIndex + state
							);
							complex<double> a0 = momentumSpaceContext.getAmplitude(
								kMinusQLinearIndex/numOrbitals,
								state,
								propagatorEnd
							);
							complex<double> a1 = momentumSpaceContext.getAmplitude(
								kMinusQLinearIndex/numOrbitals,
								state,
								propagatorStart
							);

							for(
								unsigned int e0 = 0;
								e0 < numSummationEnergies;
								e0++
							){
								zFactor[
									numOrbitals*incommingIndex
									+ outgoingIndex
								] -= selfEnergyVertex.at(e0)*a0*conj(a1)/pow(
									summationEnergies.at(e0)
									- e
									+ model.getChemicalPotential()
									+ i*eta,
									2
								);
							}

//							complex<double> prefactor = conj(a0)*a1*interactionAmplitude0.getAmplitude()*interactionAmplitude1.getAmplitude();
//							for(int j = 0; j < numMatsubaraFrequencies; j++){
//								matrix[numOrbitals*incommingIndex + outgoingIndex] += prefactor*susceptibility.at(j)/pow(context.susceptibilityEnergies.at(j) - e + model->getChemicalPotential(), 2);
//							}
						}
					}
				}
			}
		}
	}

	for(unsigned int n = 0; n < numOrbitals*numOrbitals; n++)
		zFactor[n] *= kT/mesh.size();

	for(unsigned int n = 0; n < numOrbitals; n++)
		zFactor[numOrbitals*n + n] += 1.;

	complex<double> *eigenVectors = new complex<double>[
		numOrbitals*numOrbitals
	];
	complex<double> *eigenVectorsHermitianConjugate = new complex<double>[
		numOrbitals*numOrbitals
	];
	for(unsigned int row = 0; row < numOrbitals; row++){
		for(unsigned int col = 0; col < numOrbitals; col++){
			complex<double> amplitude = propertyExtractor.getAmplitude(
				kIndex,
				row,
				{(int)col}
			);

			eigenVectors[numOrbitals*col + row] = amplitude;
			eigenVectorsHermitianConjugate[
				numOrbitals*row + col
			] = conj(amplitude);
		}
	}

	complex<double> *zFactorWorkspace = new complex<double>[
		numOrbitals*numOrbitals
	];
	multiplyMatrices(eigenVectors, zFactor, zFactorWorkspace, numOrbitals);
	multiplyMatrices(
		zFactorWorkspace,
		eigenVectorsHermitianConjugate,
		zFactor,
		numOrbitals
	);
	invertMatrix(zFactor, numOrbitals);

	vector<complex<double>> zFactors;
	for(unsigned int n = 0; n < numOrbitals; n++)
		zFactors.push_back(zFactor[numOrbitals*n + n]);

	delete [] zFactor;
	delete [] zFactorWorkspace;
	delete [] eigenVectors;
	delete [] eigenVectorsHermitianConjugate;

//	for(unsigned int n = 0; n < numOrbitals*numOrbitals; n++)
//		matrix[n] = -matrix[n]*kT/(double)mesh.size();
//	for(unsigned int n = 0; n < numOrbitals; n++)
//		matrix[n*numOrbitals + n] += 1.;

//	invertMatrix(matrix, numOrbitals);

//	return abs(matrix[0]) + abs(matrix[1]) + abs(matrix[2]) + abs(matrix[3]);
//	return abs(matrix[3]);
	return zFactors;
}*/

/*complex<double> SusceptibilityCalculator::calculateZFactor(
	const vector<double> &k,
	Context &context
){
	Streams::out << counter << "\n";
	TBTKAssert(
		isInitialized,
		"SusceptibilityCalculator::calculateSusceptibility()",
		"SusceptibilityCalculator not yet initialized.",
		"Use SusceptibilityCalculator::init() to initialize the"
		<< " SusceptibilityCalculator."
	);

	generateKPlusQLookupTable();

	//Basis size
	int basisSize = model->getBasisSize();

	complex<double> matrix[numOrbitals*numOrbitals];
	for(unsigned int n = 0; n < numOrbitals*numOrbitals; n++)
		matrix[n] = 0.;

//	Timer::tick("Main loop");
	int numMatsubaraFrequencies = 50;
	double t = UnitHandler::convertTemperatureNtB(model->getTemperature());
	double kT = UnitHandler::getK_BB()*t;

	if(context.susceptibilityEnergies.size() == 0){
		for(int j = -numMatsubaraFrequencies/2; j < numMatsubaraFrequencies/2; j++)
			context.susceptibilityEnergies.push_back(M_PI*(1 + 2*j)*kT*i);
		context.setSusceptibilityEnergiesAreInversionSymmetric(true);
	}
	context.setSusceptibilityIsSafeFromPoles(true);

	//Main loop
	for(unsigned int n = 0; n < mesh.size(); n++){
		Index kMinusQIndex = brillouinZone->getMinorCellIndex(
			{mesh[n][0] - k[0], mesh[n][1] - k[1]},
			numMeshPoints
		);
		int kMinusQLinearIndex =  model->getHoppingAmplitudeSet()->getFirstIndexInBlock(kMinusQIndex);
		for(unsigned int c = 0; c < interactionAmplitudes.size(); c++){
			const InteractionAmplitude interactionAmplitude0 = interactionAmplitudes.at(c);
			int propagatorIndex0 = interactionAmplitude0.getCreationOperatorIndex(0).at(0);
			int susceptibilityIndex2 = interactionAmplitude0.getCreationOperatorIndex(1).at(0);
			int susceptibilityIndex3 = interactionAmplitude0.getAnnihilationOperatorIndex(0).at(0);
			int incommingIndex = interactionAmplitude0.getAnnihilationOperatorIndex(1).at(0);

			for(unsigned int d = 0; d < interactionAmplitudes.size(); d++){
				const InteractionAmplitude interactionAmplitude1 = interactionAmplitudes.at(d);
				int outgoingIndex = interactionAmplitude1.getCreationOperatorIndex(0).at(0);
				int susceptibilityIndex0 = interactionAmplitude1.getCreationOperatorIndex(1).at(0);
				int susceptibilityIndex1 = interactionAmplitude1.getAnnihilationOperatorIndex(0).at(0);
				int propagatorIndex1 = interactionAmplitude1.getAnnihilationOperatorIndex(1).at(0);

				vector<complex<double>> susceptibility = calculateRPASusceptibility(
					mesh.at(n),
					{
						susceptibilityIndex0,
						susceptibilityIndex1,
						susceptibilityIndex2,
						susceptibilityIndex3
					},
					context
				);

				for(unsigned int m = 0; m < numOrbitals; m++){
					double e = energies[(kMinusQLinearIndex + basisSize + (int)m)%basisSize];
					complex<double> a0 = amplitudes[numOrbitals*((kMinusQLinearIndex + basisSize + m)%basisSize) + propagatorIndex0];
					complex<double> a1 = amplitudes[numOrbitals*((kMinusQLinearIndex + basisSize + m)%basisSize) + propagatorIndex1];

					complex<double> prefactor = conj(a0)*a1*interactionAmplitude0.getAmplitude()*interactionAmplitude1.getAmplitude();
					for(int j = 0; j < numMatsubaraFrequencies; j++){
						matrix[numOrbitals*incommingIndex + outgoingIndex] += prefactor*susceptibility.at(j)/pow(context.susceptibilityEnergies.at(j) - e + model->getChemicalPotential(), 2);
					}
				}
			}
		}
	}

	for(unsigned int n = 0; n < numOrbitals*numOrbitals; n++)
		matrix[n] = -matrix[n]*kT/(double)mesh.size();
	for(unsigned int n = 0; n < numOrbitals; n++)
		matrix[n*numOrbitals + n] += 1.;

	invertMatrix(matrix, numOrbitals);

//	return abs(matrix[0]) + abs(matrix[1]) + abs(matrix[2]) + abs(matrix[3]);
	return abs(matrix[3]);
}*/

}	//End of namesapce TBTK
