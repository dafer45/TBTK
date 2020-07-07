/* Copyright 2016 Kristofer Björnson
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

/** @file ChebyshevExpander.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Solver/ChebyshevExpander.h"
#include "TBTK/HALinkedList.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/UnitHandler.h"

#include <iostream>
#include <cmath>

using namespace std;

namespace TBTK{
namespace Solver{

DynamicTypeInformation ChebyshevExpander::dynamicTypeInformation(
	"Solver::ChebyshevExpander",
	{&Solver::dynamicTypeInformation}
);

namespace{
	const complex<double> i(0, 1);
}

ChebyshevExpander::ChebyshevExpander() : Communicator(false){
	scaleFactor = 1.1;
	numCoefficients = 1000;
	broadening = 1e-6;
	energyWindow = Range(-1, 1, 1000);
	calculateCoefficientsOnGPU = false;
	generateGreensFunctionsOnGPU = false;
	useLookupTable = false;
	generatingFunctionLookupTable.setIsValid(false);
	generatingFunctionLookupTable_device = NULL;
	lookupTableNumCoefficients = 0;
	lookupTableResolution = 0;
	lookupTableLowerBound = 0.;
	lookupTableUpperBound = 0.;
}

ChebyshevExpander::~ChebyshevExpander(){
	if(generatingFunctionLookupTable_device != nullptr)
		destroyLookupTableGPU();
}

void addHamiltonianProduct(
	const SparseMatrix<complex<double>> &sparseMatrix,
	const CArray<complex<double>> &vector,
	CArray<complex<double>> &result
){
	const unsigned int *csrRowPointers = sparseMatrix.getCSRRowPointers();
	const unsigned int *csrColumns = sparseMatrix.getCSRColumns();
	const complex<double> *values = sparseMatrix.getCSRValues();
	for(unsigned int row = 0; row < sparseMatrix.getNumRows(); row++){
		for(
			unsigned int n = csrRowPointers[row];
			n < csrRowPointers[row+1];
			n++
		){
			result[row] += values[n]*vector[csrColumns[n]];
		}
	}
}

void cyclicSwap(
	CArray<complex<double>> &jIn1,
	CArray<complex<double>> &jIn2,
	CArray<complex<double>> &jResult
){
	CArray<complex<double>> temp = std::move(jIn2);
	jIn2 = std::move(jIn1);
	jIn1 = std::move(jResult);
	jResult = std::move(temp);
}

vector<complex<double>> ChebyshevExpander::calculateCoefficientsCPU(
	Index to,
	Index from
){
	TBTKAssert(
		scaleFactor > 0,
		"ChebyshevExpander::calculateCoefficients()",
		"Scale factor must be larger than zero.",
		"Use ChebyshevExpander::setScaleFactor() to set scale factor."
	);
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevExpander::calculateCoefficients()",
		"numCoefficients has to be larger than 0.",
		""
	);

	vector<complex<double>> coefficients;
	coefficients.reserve(numCoefficients);
	for(int n = 0; n < numCoefficients; n++)
		coefficients.push_back(0);

	const HoppingAmplitudeSet &hoppingAmplitudeSet
		= getModel().getHoppingAmplitudeSet();
	const int basisSize = hoppingAmplitudeSet.getBasisSize();

	int fromBasisIndex = hoppingAmplitudeSet.getBasisIndex(from);
	int toBasisIndex = hoppingAmplitudeSet.getBasisIndex(to);

	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "ChebyshevExpander::calculateCoefficients\n";
		Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Streams::out << "\tTo Index: " << toBasisIndex << "\n";
		Streams::out << "\tBasis size: " << basisSize << "\n";
		Streams::out << "\tProgress (100 coefficients per dot): ";
	}

	CArray<complex<double>> jIn1(basisSize);
	CArray<complex<double>> jIn2(basisSize);
	CArray<complex<double>> jResult(basisSize);
	for(int n = 0; n < basisSize; n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	coefficients[0] = jIn1[toBasisIndex];

	//Get the Hamiltonian on SparseMatrix format and scale it by the scale
	//factor.
	SparseMatrix<complex<double>> sparseMatrix
		= hoppingAmplitudeSet.getSparseMatrix();
	sparseMatrix.setStorageFormat(
		SparseMatrix<complex<double>>::StorageFormat::CSR
	);
	sparseMatrix *= 1/scaleFactor;

	//Calculate |j1>
	for(int c = 0; c < basisSize; c++)
		jResult[c] = 0.;
	addHamiltonianProduct(sparseMatrix, jIn1, jResult);
	cyclicSwap(jIn1, jIn2, jResult);
	coefficients[1] = jIn1[toBasisIndex];

	//Multiply hopping amplitudes by factor two, to speed up calculation of
	//the first term in 2H|j(n-1)> - |j(n-2)>.
	sparseMatrix *= 2;

	//Iteratively calculate |jn> and corresponding Chebyshev coefficients.
	for(int n = 2; n < numCoefficients; n++){
		for(int c = 0; c < basisSize; c++)
			jResult[c] = -jIn2[c];
		addHamiltonianProduct(sparseMatrix, jIn1, jResult);
		cyclicSwap(jIn1, jIn2, jResult);
		coefficients[n] = jIn1[toBasisIndex];

		if(getGlobalVerbose() && getVerbose()){
			if(n%100 == 0)
				Streams::out << "." << flush;
			if(n%1000 == 0)
				Streams::out << " " << flush;
		}
	}
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\n";

	//Lorentzian convolution
	if(broadening != 0){
		double lambda = broadening*numCoefficients;
		for(int n = 0; n < numCoefficients; n++)
			coefficients[n] = coefficients[n]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
	}

	return coefficients;
}

vector<vector<complex<double>>> ChebyshevExpander::calculateCoefficientsCPU(
	vector<Index> &to,
	Index from
){
	TBTKAssert(
		scaleFactor > 0,
		"ChebyshevExpander::calculateCoefficients()",
		"Scale factor must be larger than zero.",
		"Use ChebyshevExpander::setScaleFactor() to set scale factor."
	);
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevExpander::calculateCoefficients()",
		"numCoefficients has to be larger than 0.",
		""
	);

	vector<vector<complex<double>>> coefficients;
	for(unsigned int n = 0; n < to.size(); n++){
		coefficients.push_back(vector<complex<double>>());
		coefficients[n].reserve(numCoefficients);
		for(int c = 0; c < numCoefficients; c++)
			coefficients[n].push_back(0);
	}

	const HoppingAmplitudeSet &hoppingAmplitudeSet
		= getModel().getHoppingAmplitudeSet();
	unsigned int basisSize = hoppingAmplitudeSet.getBasisSize();

	int fromBasisIndex = hoppingAmplitudeSet.getBasisIndex(from);
	CArray<int> coefficientMap(basisSize);
	for(unsigned int n = 0; n < basisSize; n++)
		coefficientMap[n] = -1;
	for(unsigned int n = 0; n < to.size(); n++)
		coefficientMap[hoppingAmplitudeSet.getBasisIndex(to.at(n))] = n;

	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "ChebyshevExpander::calculateCoefficients\n";
		Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Streams::out << "\tBasis size: " << basisSize << "\n";
		Streams::out << "\tProgress (100 coefficients per dot): ";
	}

	CArray<complex<double>> jIn1(basisSize);
	CArray<complex<double>> jIn2(basisSize);
	CArray<complex<double>> jResult(basisSize);
	for(unsigned int n = 0; n < basisSize; n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	for(unsigned int n = 0; n < basisSize; n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]][0] = jIn1[n];

	//Get the Hamiltonian on SparseMatrix format and scale it by the scale
	//factor.
	SparseMatrix<complex<double>> sparseMatrix
		= hoppingAmplitudeSet.getSparseMatrix();
	sparseMatrix.setStorageFormat(
		SparseMatrix<complex<double>>::StorageFormat::CSR
	);
	sparseMatrix *= 1/scaleFactor;

	//Calculate |j1>
	for(unsigned int c = 0; c < basisSize; c++)
		jResult[c] = 0.;
	addHamiltonianProduct(sparseMatrix, jIn1, jResult);
	cyclicSwap(jIn1, jIn2, jResult);
	for(unsigned int n = 0; n < basisSize; n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]][1] = jIn1[n];

	//Multiply hopping amplitudes by factor two, to speed up calculation of
	//the first term in 2H|j(n-1)> - |j(n-2)>.
	sparseMatrix *= 2;

	//Iteratively calculate |jn> and corresponding Chebyshev coefficients.
	for(int n = 2; n < numCoefficients; n++){
		for(unsigned int c = 0; c < basisSize; c++)
			jResult[c] = -jIn2[c];
		addHamiltonianProduct(sparseMatrix, jIn1, jResult);
		cyclicSwap(jIn1, jIn2, jResult);
		for(unsigned int c = 0; c < basisSize; c++)
			if(coefficientMap[c] != -1)
				coefficients[coefficientMap[c]][n] = jIn1[c];

		if(getGlobalVerbose() && getVerbose()){
			if(n%100 == 0)
				Streams::out << "." << flush;
			if(n%1000 == 0)
				Streams::out << " " << flush;
		}
	}
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\n";

	//Lorentzian convolution
	if(broadening != 0){
		double lambda = broadening*numCoefficients;
		for(int n = 0; n < numCoefficients; n++)
			for(unsigned int c = 0; c < to.size(); c++)
				coefficients[c][n] = coefficients[c][n]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
	}

	return coefficients;
}

void ChebyshevExpander::generateLookupTable(){
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevExpander::generateLookupTable()",
		"numCoefficients has to be larger than 0.",
		""
	);
	TBTKAssert(
		energyWindow.getResolution() > 0,
		"ChebyshevExpander::generateLookupTable()",
		"The energy windows resolution has to be larger than 0.",
		""
	);
	TBTKAssert(
		energyWindow[0] < energyWindow.getLast(),
		"ChebyshevExpander::generateLookupTable()",
		"The energy windows lower bound has to be smaller than its"
		<< " upper bound.",
		""
	);
	TBTKAssert(
		energyWindow[0] > -scaleFactor,
		"ChebyshevExpander::generateLookupTable()",
		"The energy windows lower bound has to be larger than"
		<< " -scaleFactor.",
		"Use ChebyshevExpander::setScaleFactor to set a larger scale"
		<< " factor."
	);
	TBTKAssert(
		energyWindow.getLast() < scaleFactor,
		"ChebyshevExpander::generateLookupTable()",
		"The energy windows upper bound has to be smaller than"
		<< " scaleFactor.",
		"Use ChebyshevExpander::setScaleFactor to set a larger scale"
		<< " factor."
	);

	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "Generating lookup table\n";
		Streams::out << "\tNum coefficients: " << numCoefficients << "\n";
		Streams::out << "\tEnergy resolution: " << energyWindow.getResolution() << "\n";
		Streams::out << "\tLower bound: " << energyWindow[0] << "\n";
		Streams::out << "\tUpper bound: " << energyWindow.getLast() << "\n";
	}

	lookupTableNumCoefficients = numCoefficients;
	lookupTableResolution = energyWindow.getResolution();
	lookupTableLowerBound = energyWindow[0];
	lookupTableUpperBound = energyWindow.getLast();

	generatingFunctionLookupTable
		= CArray<CArray<complex<double>>>(numCoefficients);
	for(int n = 0; n < numCoefficients; n++){
		generatingFunctionLookupTable[n]
			= CArray<complex<double>>(energyWindow.getResolution());
	}

	#pragma omp parallel for
	for(int n = 0; n < numCoefficients; n++){
		double denominator = 1.;
		if(n == 0)
			denominator = 2.;

		for(unsigned int e = 0; e < energyWindow.getResolution(); e++){
			double E = energyWindow[e]/scaleFactor;
			generatingFunctionLookupTable[n][e] = (1/scaleFactor)*(-2.*i/sqrt(1 - E*E))*exp(-i*((double)n)*acos(E))/denominator;
		}
	}
	generatingFunctionLookupTable.setIsValid(true);
}

void ChebyshevExpander::destroyLookupTable(){
	generatingFunctionLookupTable = CArray<CArray<complex<double>>>();
	generatingFunctionLookupTable.setIsValid(false);
}

vector<complex<double>> ChebyshevExpander::generateGreensFunctionCPU(
	const vector<complex<double>> &coefficients,
	Type type
){
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevExpander::generateGreensFunctionCPU()",
		"numCoefficients has to be larger than 0.",
		""
	);
	TBTKAssert(
		energyWindow.getResolution() > 0,
		"ChebyshevExpander::generateGreensFunctionCPU()",
		"The energy resolution has to be larger than 0.",
		""
	);
	TBTKAssert(
		energyWindow[0] < energyWindow.getLast(),
		"ChebyshevExpander::generateGreensFunctionCPU()",
		"The energy windows lower bound has to be smaller than upper"
		<< " bound.",
		""
	);
	TBTKAssert(
		energyWindow[0] > -scaleFactor,
		"ChebyshevExpander::generateGreensFunctionCPU()",
		"The energy windows lower bound has to be larger than"
		<< " -scaleFactor.",
		"Use ChebyshevExpander::setScaleFactor to set a larger scale"
		<< " factor."
	);
	TBTKAssert(
		energyWindow.getLast() < scaleFactor,
		"ChebyshevExpander::generateGreensFunctionCPU()",
		"The energy windows upper bound has to be smaller than"
		<< " scaleFactor.",
		"Use ChebyshevExpander::setScaleFactor to set a larger scale"
		<< " factor."
	);

	ensureLookupTableIsReady();

	vector<complex<double>> greensFunctionData(
		energyWindow.getResolution(),
		0
	);

	if(type == Type::Retarded){
		if(useLookupTable){
			for(int n = 0; n < lookupTableNumCoefficients; n++){
				for(int e = 0; e < lookupTableResolution; e++){
					greensFunctionData[e] += generatingFunctionLookupTable[n][e]*coefficients[n];
				}
			}
		}
		else{
			for(int n = 0; n < numCoefficients; n++){
				double denominator = 1.;
				if(n == 0)
					denominator = 2.;

				for(
					unsigned int e = 0;
					e < energyWindow.getResolution();
					e++
				){
					double E = energyWindow[e]/scaleFactor;
					greensFunctionData[e] += coefficients[n]*(1/scaleFactor)*(-2.*i/sqrt(1 - E*E))*exp(-i*((double)n)*acos(E))/denominator;
				}
			}
		}
	}
	else if(type == Type::Advanced){
		if(useLookupTable){
			for(int n = 0; n < lookupTableNumCoefficients; n++){
				for(int e = 0; e < lookupTableResolution; e++){
					greensFunctionData[e] += coefficients[n]*conj(generatingFunctionLookupTable[n][e]);
				}
			}
		}
		else{
			for(int n = 0; n < numCoefficients; n++){
				double denominator = 1.;
				if(n == 0)
					denominator = 2.;

				for(
					unsigned int e = 0;
					e < energyWindow.getResolution();
					e++
				){
					double E = energyWindow[e]/scaleFactor;
					greensFunctionData[e] += coefficients[n]*conj((1/scaleFactor)*(-2.*i/sqrt(1 - E*E))*exp(-i*((double)n)*acos(E))/denominator);
				}
			}
		}
	}
	else if(type == Type::Principal){
		if(useLookupTable){
			for(int n = 0; n < lookupTableNumCoefficients; n++){
				for(int e = 0; e < lookupTableResolution; e++){
					greensFunctionData[e] += -coefficients[n]*real(generatingFunctionLookupTable[n][e]);
				}
			}
		}
		else{
			for(int n = 0; n < numCoefficients; n++){
				double denominator = 1.;
				if(n == 0)
					denominator = 2.;

				for(
					unsigned int e = 0;
					e < energyWindow.getResolution();
					e++
				){
					double E = energyWindow[e]/scaleFactor;
					greensFunctionData[e] += coefficients[n]*(1/scaleFactor)*(2./sqrt(1 - E*E))*sin(((double)n)*acos(E))/denominator;
				}
			}
		}
	}
	else if(type == Type::NonPrincipal){
		if(useLookupTable){
			for(int n = 0; n < lookupTableNumCoefficients; n++){
				for(int e = 0; e < lookupTableResolution; e++){
					greensFunctionData[e] -= coefficients[n]*i*imag(generatingFunctionLookupTable[n][e]);
				}
			}
		}
		else{
			for(int n = 0; n < numCoefficients; n++){
				double denominator = 1.;
				if(n == 0)
					denominator = 2.;

				for(
					unsigned int e = 0;
					e < energyWindow.getResolution();
					e++
				){
					double E = energyWindow[e]/scaleFactor;
					greensFunctionData[e] += coefficients[n]*(1/scaleFactor)*(2.*i/sqrt(1 - E*E))*cos(((double)n)*acos(E))/denominator;
				}
			}
		}
	}
	else{
		TBTKExit(
			"ChebyshevExpander::generateGreensFunction()",
			"Unknown GreensFunctionType",
			""
		);
	}

	return greensFunctionData;
}

};	//End of namespace Solver
};	//End of namespace TBTK
