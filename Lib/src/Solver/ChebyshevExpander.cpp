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

namespace{
	const complex<double> i(0, 1);
}

ChebyshevExpander::ChebyshevExpander() : Communicator(false){
	scaleFactor = 1.;
	numCoefficients = 1000;
	broadening = 1e-6;
	energyResolution = 1000;
	lowerBound = -1;
	upperBound = 1;
	calculateCoefficientsOnGPU = false;
	generateGreensFunctionsOnGPU = false;
	useLookupTable = false;
	damping = NULL;
	generatingFunctionLookupTable = NULL;
	generatingFunctionLookupTable_device = NULL;
	lookupTableNumCoefficients = 0;
	lookupTableResolution = 0;
	lookupTableLowerBound = 0.;
	lookupTableUpperBound = 0.;
}

ChebyshevExpander::~ChebyshevExpander(){
	if(generatingFunctionLookupTable != nullptr){
		for(int n = 0; n < lookupTableNumCoefficients; n++)
			delete [] generatingFunctionLookupTable[n];

		delete [] generatingFunctionLookupTable;
	}
	if(generatingFunctionLookupTable_device != nullptr)
		destroyLookupTableGPU();
}

vector<complex<double>> ChebyshevExpander::calculateCoefficientsCPU(
	Index to,
	Index from
){
	const Model &model = getModel();

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

	const HoppingAmplitudeSet &hoppingAmplitudeSet = model.getHoppingAmplitudeSet();

	int fromBasisIndex = hoppingAmplitudeSet.getBasisIndex(from);
	int toBasisIndex = hoppingAmplitudeSet.getBasisIndex(to);

	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "ChebyshevExpander::calculateCoefficients\n";
		Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Streams::out << "\tTo Index: " << toBasisIndex << "\n";
		Streams::out << "\tBasis size: " << hoppingAmplitudeSet.getBasisSize() << "\n";
		Streams::out << "\tProgress (100 coefficients per dot): ";
	}

	complex<double> *jIn1 = new complex<double>[hoppingAmplitudeSet.getBasisSize()];
	complex<double> *jIn2 = new complex<double>[hoppingAmplitudeSet.getBasisSize()];
	complex<double> *jResult = new complex<double>[hoppingAmplitudeSet.getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	coefficients[0] = jIn1[toBasisIndex];

	//Generate a fixed hopping amplitude and inde list, for speed.
	int numHoppingAmplitudes = 0;
	for(
		HoppingAmplitudeSet::ConstIterator iterator
			= hoppingAmplitudeSet.cbegin();
		iterator != hoppingAmplitudeSet.cend();
		++iterator
	){
		numHoppingAmplitudes++;
	}

	complex<double> *hoppingAmplitudes = new complex<double>[numHoppingAmplitudes];
	int *toIndices = new int[numHoppingAmplitudes];
	int *fromIndices = new int[numHoppingAmplitudes];
	int counter = 0;
	for(
		HoppingAmplitudeSet::ConstIterator iterator
			= hoppingAmplitudeSet.cbegin();
		iterator != hoppingAmplitudeSet.cend();
		++iterator
	){
		toIndices[counter] = hoppingAmplitudeSet.getBasisIndex((*iterator).getToIndex());
		fromIndices[counter] = hoppingAmplitudeSet.getBasisIndex((*iterator).getFromIndex());
		hoppingAmplitudes[counter] = (*iterator).getAmplitude()/scaleFactor;

		counter++;
	}

	//Calculate |j1>
	for(int c = 0; c < hoppingAmplitudeSet.getBasisSize(); c++)
		jResult[c] = 0.;
	for(int n = 0; n < numHoppingAmplitudes; n++){
		int from = fromIndices[n];
		int to = toIndices[n];

		jResult[to] += hoppingAmplitudes[n]*jIn1[from];
	}

	if(damping != NULL){
		for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++)
			jResult[n] *= damping[n];
	}

	jTemp = jIn2;
	jIn2 = jIn1;
	jIn1 = jResult;
	jResult = jTemp;

	coefficients[1] = jIn1[toBasisIndex];

	//Multiply hopping amplitudes by factor two, to spped up calculation of 2H|j(n-1)> - |j(n-2)>.
	for(int n = 0; n < numHoppingAmplitudes; n++)
		hoppingAmplitudes[n] *= 2.;

	//Iteratively calculate |jn> and corresponding Chebyshev coefficients.
	for(int n = 2; n < numCoefficients; n++){
		for(int c = 0; c < hoppingAmplitudeSet.getBasisSize(); c++)
			jResult[c] = -jIn2[c];

		if(damping != NULL){
			for(int c = 0; c < hoppingAmplitudeSet.getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		for(int c = 0; c < numHoppingAmplitudes; c++){
			int from = fromIndices[c];
			int to = toIndices[c];

			jResult[to] += hoppingAmplitudes[c]*jIn1[from];
		}

		if(damping != NULL){
			for(int c = 0; c < hoppingAmplitudeSet.getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		jTemp = jIn2;
		jIn2 = jIn1;
		jIn1 = jResult;
		jResult = jTemp;

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

	delete [] jIn1;
	delete [] jIn2;
	delete [] jResult;
	delete [] hoppingAmplitudes;
	delete [] toIndices;
	delete [] fromIndices;

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
	const Model &model = getModel();
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

	const HoppingAmplitudeSet &hoppingAmplitudeSet = model.getHoppingAmplitudeSet();

	int fromBasisIndex = hoppingAmplitudeSet.getBasisIndex(from);
	int *coefficientMap = new int[hoppingAmplitudeSet.getBasisSize()];
	for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++)
		coefficientMap[n] = -1;
	for(unsigned int n = 0; n < to.size(); n++)
		coefficientMap[hoppingAmplitudeSet.getBasisIndex(to.at(n))] = n;

	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "ChebyshevExpander::calculateCoefficients\n";
		Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Streams::out << "\tBasis size: " << hoppingAmplitudeSet.getBasisSize() << "\n";
		Streams::out << "\tProgress (100 coefficients per dot): ";
	}

	complex<double> *jIn1 = new complex<double>[hoppingAmplitudeSet.getBasisSize()];
	complex<double> *jIn2 = new complex<double>[hoppingAmplitudeSet.getBasisSize()];
	complex<double> *jResult = new complex<double>[hoppingAmplitudeSet.getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]][0] = jIn1[n];
//			coefficients[coefficientMap[n]*numCoefficients] = jIn1[n];

	//Generate a fixed hopping amplitude and inde list, for speed.
	int numHoppingAmplitudes = 0;
	for(
		HoppingAmplitudeSet::ConstIterator iterator
			= hoppingAmplitudeSet.cbegin();
		iterator != hoppingAmplitudeSet.cend();
		++iterator
	){
		numHoppingAmplitudes++;
	}

	complex<double> *hoppingAmplitudes = new complex<double>[numHoppingAmplitudes];
	int *toIndices = new int[numHoppingAmplitudes];
	int *fromIndices = new int[numHoppingAmplitudes];
	int counter = 0;
	for(
		HoppingAmplitudeSet::ConstIterator iterator
			= hoppingAmplitudeSet.cbegin();
		iterator != hoppingAmplitudeSet.cend();
		++iterator
	){
		toIndices[counter] = hoppingAmplitudeSet.getBasisIndex((*iterator).getToIndex());
		fromIndices[counter] = hoppingAmplitudeSet.getBasisIndex((*iterator).getFromIndex());
		hoppingAmplitudes[counter] = (*iterator).getAmplitude()/scaleFactor;

		counter++;
	}

	//Calculate |j1>
	for(int c = 0; c < hoppingAmplitudeSet.getBasisSize(); c++)
		jResult[c] = 0.;
	for(int n = 0; n < numHoppingAmplitudes; n++){
		int from = fromIndices[n];
		int to = toIndices[n];

		jResult[to] += hoppingAmplitudes[n]*jIn1[from];
	}

	if(damping != NULL){
		for(int c = 0; c < hoppingAmplitudeSet.getBasisSize(); c++)
			jResult[c] *= damping[c];
	}

	jTemp = jIn2;
	jIn2 = jIn1;
	jIn1 = jResult;
	jResult = jTemp;

	for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]][1] = jIn1[n];
//			coefficients[coefficientMap[n]*numCoefficients + 1] = jIn1[n];

	//Multiply hopping amplitudes by factor two, to spped up calculation of 2H|j(n-1)> - |j(n-2)>.
	for(int n = 0; n < numHoppingAmplitudes; n++)
		hoppingAmplitudes[n] *= 2.;

	//Iteratively calculate |jn> and corresponding Chebyshev coefficients.
	for(int n = 2; n < numCoefficients; n++){
		for(int c = 0; c < hoppingAmplitudeSet.getBasisSize(); c++)
			jResult[c] = -jIn2[c];

		if(damping != NULL){
			for(int c = 0; c < hoppingAmplitudeSet.getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		for(int c = 0; c < numHoppingAmplitudes; c++){
			int from = fromIndices[c];
			int to = toIndices[c];

			jResult[to] += hoppingAmplitudes[c]*jIn1[from];
		}

		if(damping != NULL){
			for(int c = 0; c < hoppingAmplitudeSet.getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		jTemp = jIn2;
		jIn2 = jIn1;
		jIn1 = jResult;
		jResult = jTemp;

		for(int c = 0; c < hoppingAmplitudeSet.getBasisSize(); c++)
			if(coefficientMap[c] != -1)
				coefficients[coefficientMap[c]][n] = jIn1[c];
//				coefficients[coefficientMap[c]*numCoefficients + n] = jIn1[c];

		if(getGlobalVerbose() && getVerbose()){
			if(n%100 == 0)
				Streams::out << "." << flush;
			if(n%1000 == 0)
				Streams::out << " " << flush;
		}
	}
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\n";

	delete [] jIn1;
	delete [] jIn2;
	delete [] jResult;
	delete [] hoppingAmplitudes;
	delete [] toIndices;
	delete [] fromIndices;

	//Lorentzian convolution
	if(broadening != 0){
		double lambda = broadening*numCoefficients;
		for(int n = 0; n < numCoefficients; n++)
			for(unsigned int c = 0; c < to.size(); c++)
				coefficients[c][n] = coefficients[c][n]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
//			coefficients[n] = coefficients[n]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
	}

	return coefficients;
}

void ChebyshevExpander::calculateCoefficientsWithCutoff(
	Index to,
	Index from,
	complex<double> *coefficients,
	int numCoefficients,
	double componentCutoff,
	double broadening
){
	const Model &model = getModel();

/*	TBTKAssert(
		model != NULL,
		"ChebyshevExpander::calculateCoefficientsWithCutoff()",
		"Model not set.",
		"Use ChebyshevExpander::setModel() to set model."
	);*/
	TBTKAssert(
		scaleFactor > 0,
		"ChebyshevExpander::calculateCoefficientsWithCutoff()",
		"Scale factor must be larger than zero.",
		"Use ChebyshevExpander::setScaleFactor() to set scale factor."
	);
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevExpander::calculateCoefficientsWithCutoff()",
		"numCoefficients has to be larger than 0.",
		""
	);

	const HoppingAmplitudeSet &hoppingAmplitudeSet = model.getHoppingAmplitudeSet();

	int fromBasisIndex = hoppingAmplitudeSet.getBasisIndex(from);
	int toBasisIndex = hoppingAmplitudeSet.getBasisIndex(to);

	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "ChebyshevExpander::calculateCoefficients\n";
		Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Streams::out << "\tTo Index: " << toBasisIndex << "\n";
		Streams::out << "\tBasis size: " << hoppingAmplitudeSet.getBasisSize() << "\n";
		Streams::out << "\tProgress (100 coefficients per dot): ";
	}

	complex<double> *jIn1 = new complex<double>[hoppingAmplitudeSet.getBasisSize()];
	complex<double> *jIn2 = new complex<double>[hoppingAmplitudeSet.getBasisSize()];
	complex<double> *jResult = new complex<double>[hoppingAmplitudeSet.getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state |j0>.
	jIn1[fromBasisIndex] = 1.;

	coefficients[0] = jIn1[toBasisIndex];

	HALinkedList haLinkedList(hoppingAmplitudeSet);
	haLinkedList.rescaleAmplitudes(scaleFactor);
	haLinkedList.addLinkedList(fromBasisIndex);

	int *newlyReachedIndices = new int[hoppingAmplitudeSet.getBasisSize()];
	int *everReachedIndices = new int[hoppingAmplitudeSet.getBasisSize()];
	bool *everReachedIndicesAdded = new bool[hoppingAmplitudeSet.getBasisSize()];
	int newlyReachedIndicesCounter = 0;
	int everReachedIndicesCounter = 0;
	for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++){
		newlyReachedIndices[n] = -1;
		everReachedIndices[n] = -1;
		everReachedIndicesAdded[n] = false;
	}

	HALink *link = haLinkedList.getFirstMainLink();
	while(link != NULL){
		int from = link->from;
		int to = link->to;

		jResult[to] += link->amplitude*jIn1[from];
		if(!everReachedIndicesAdded[to]){
			newlyReachedIndices[newlyReachedIndicesCounter] = to;
			newlyReachedIndicesCounter++;
		}

		link = link->next2;
	}

	for(int n = 0; n < newlyReachedIndicesCounter; n++){
		if(abs(jResult[newlyReachedIndices[n]]) > componentCutoff){
			haLinkedList.addLinkedList(newlyReachedIndices[n]);
			if(!everReachedIndicesAdded[newlyReachedIndices[n]]){
				everReachedIndicesAdded[newlyReachedIndices[n]] = true;
				everReachedIndices[everReachedIndicesCounter] = newlyReachedIndices[n];
				everReachedIndicesCounter++;
			}
		}
		else{
			jResult[newlyReachedIndices[n]] = 0.;
		}
	}

	jTemp = jIn2;
	jIn2 = jIn1;
	jIn1 = jResult;
	jResult = jTemp;

	coefficients[1] = jIn1[toBasisIndex];

	//Multiply hopping amplitudes by factor two, to speed up calculation of 2H|j(n-1)> - |j(n-2)>.
	HALink *links = haLinkedList.getLinkArray();
	for(int n = 0; n < haLinkedList.getLinkArraySize(); n++)
		links[n].amplitude *= 2.;

	//Iteratively calcuate |jn> and corresponding coefficients.
	for(int n = 2; n < numCoefficients; n++){
		for(int c = 0; c < everReachedIndicesCounter; c++)
			jResult[everReachedIndices[c]] = -jIn2[everReachedIndices[c]];

		newlyReachedIndicesCounter = 0.;
		HALink *link = haLinkedList.getFirstMainLink();
		while(link != NULL){
			int from = link->from;
			int to = link->to;

			jResult[to] += link->amplitude*jIn1[from];
			if(!everReachedIndicesAdded[to]){
				newlyReachedIndices[newlyReachedIndicesCounter] = to;
				newlyReachedIndicesCounter++;
			}

			link = link->next2;
		}

		for(int c = 0; c < newlyReachedIndicesCounter; c++){
			if(abs(jResult[newlyReachedIndices[c]]) > componentCutoff){
				haLinkedList.addLinkedList(newlyReachedIndices[c]);
				if(!everReachedIndicesAdded[newlyReachedIndices[c]]){
					everReachedIndicesAdded[newlyReachedIndices[c]] = true;
					everReachedIndices[everReachedIndicesCounter] = newlyReachedIndices[c];
					everReachedIndicesCounter++;
				}
				else{
					jResult[newlyReachedIndices[c]] = 0.;
				}
			}
		}

		jTemp = jIn2;
		jIn2 = jIn1;
		jIn1 = jResult;
		jResult = jTemp;

		coefficients[n] = jIn1[toBasisIndex];

		if(getVerbose() && getGlobalVerbose()){
			if(n%100 == 0)
				Streams::out << ".";
			if(n%1000 == 0)
				Streams::out << " ";
		}
	}

	delete [] jIn1;
	delete [] jIn2;
	delete [] jResult;
	delete [] newlyReachedIndices;
	delete [] everReachedIndices;

	//Lorentzian convolution
	if(broadening != 0){
		double lambda = broadening*numCoefficients;
		for(int n = 0; n < numCoefficients; n++)
			coefficients[n] = coefficients[n]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
	}
}

void ChebyshevExpander::generateLookupTable(
	int numCoefficients,
	int energyResolution,
	double lowerBound,
	double upperBound
){
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevExpander::generateLookupTable()",
		"numCoefficients has to be larger than 0.",
		""
	);
	TBTKAssert(
		energyResolution > 0,
		"ChebyshevExpander::generateLookupTable()",
		"energyResolution has to be larger than 0.",
		""
	);
	TBTKAssert(
		lowerBound < upperBound,
		"ChebyshevExpander::generateLookupTable()",
		"lowerBound has to be smaller than upperBound.",
		""
	);
	TBTKAssert(
		lowerBound >= -scaleFactor,
		"ChebyshevExpander::generateLookupTable()",
		"lowerBound has to be larger than -scaleFactor.",
		"Use ChebyshevExpander::setScaleFactor to set a larger scale factor."
	);
	TBTKAssert(
		upperBound <= scaleFactor,
		"ChebyshevExpander::generateLookupTable()",
		"upperBound has to be smaller than scaleFactor.",
		"Use ChebyshevExpander::setScaleFactor to set a larger scale factor."
	);

	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "Generating lookup table\n";
		Streams::out << "\tNum coefficients: " << numCoefficients << "\n";
		Streams::out << "\tEnergy resolution: " << energyResolution << "\n";
		Streams::out << "\tLower bound: " << lowerBound << "\n";
		Streams::out << "\tUpper bound: " << upperBound << "\n";
	}

	if(generatingFunctionLookupTable != NULL){
		for(int n = 0; n < lookupTableNumCoefficients; n++)
			delete [] generatingFunctionLookupTable[n];

		delete [] generatingFunctionLookupTable;
	}

	lookupTableNumCoefficients = numCoefficients;
	lookupTableResolution = energyResolution;
	lookupTableLowerBound = lowerBound;
	lookupTableUpperBound = upperBound;

	generatingFunctionLookupTable = new complex<double>*[numCoefficients];
	for(int n = 0; n < numCoefficients; n++)
		generatingFunctionLookupTable[n] = new complex<double>[energyResolution];

	const double DELTA = 0.0001;
	#pragma omp parallel for
	for(int n = 0; n < numCoefficients; n++){
		double denominator = 1.;
		if(n == 0)
			denominator = 2.;

		for(int e = 0; e < energyResolution; e++){
			double E = (lowerBound + (upperBound - lowerBound)*e/(double)energyResolution)/scaleFactor;
			generatingFunctionLookupTable[n][e] = (1/scaleFactor)*(-2.*i/sqrt(1+DELTA - E*E))*exp(-i*((double)n)*acos(E))/denominator;
		}
	}
}

void ChebyshevExpander::destroyLookupTable(){
	TBTKAssert(
		generatingFunctionLookupTable != nullptr,
		"ChebyshevExpander::destroyLookupTable()",
		"No lookup table generated.",
		""
	);
	for(int n = 0; n < lookupTableNumCoefficients; n++)
		delete [] generatingFunctionLookupTable[n];

	delete [] generatingFunctionLookupTable;

	generatingFunctionLookupTable = nullptr;
}

//Property::GreensFunction* ChebyshevExpander::generateGreensFunction(
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
		energyResolution > 0,
		"ChebyshevExpander::generateGreensFunctionCPU()",
		"energyResolution has to be larger than 0.",
		""
	);
	TBTKAssert(
		lowerBound < upperBound,
		"ChebyshevExpander::generateGreensFunctionCPU()",
		"lowerBound has to be smaller than upperBound.",
		""
	);
	TBTKAssert(
		lowerBound >= -scaleFactor,
		"ChebyshevExpander::generateGreensFunctionCPU()",
		"lowerBound has to be larger than -scaleFactor.",
		"Use ChebyshevExpander::setScaleFactor to set a larger scale factor."
	);
	TBTKAssert(
		upperBound <= scaleFactor,
		"ChebyshevExpander::generateGreensFunctionCPU()",
		"upperBound has to be smaller than scaleFactor.",
		"Use ChebyshevExpander::setScaleFactor to set a larger scale factor."
	);
/*	TBTKAssert(
		generatingFunctionLookupTable != nullptr,
		"ChebyshevExpander::generateGreensFunction()",
		"Lookup table has not been generated.",
		"Use ChebyshevExpander::generateLookupTable() to generate lookup table."
	);*/

	ensureLookupTableIsReady();

/*	complex<double> *greensFunctionData = new complex<double>[energyResolution];
	for(int e = 0; e < energyResolution; e++)
		greensFunctionData[e] = 0.;*/

	vector<complex<double>> greensFunctionData(energyResolution, 0);

	const double DELTA = 0.0001;
	if(type == Type::Retarded){
		if(generatingFunctionLookupTable){
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

				for(int e = 0; e < energyResolution; e++){
					double E = (lowerBound + (upperBound - lowerBound)*e/(double)energyResolution)/scaleFactor;
					greensFunctionData[e] += coefficients[n]*(1/scaleFactor)*(-2.*i/sqrt(1+DELTA - E*E))*exp(-i*((double)n)*acos(E))/denominator;
				}
			}
		}
	}
	else if(type == Type::Advanced){
		if(generatingFunctionLookupTable){
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

				for(int e = 0; e < energyResolution; e++){
					double E = (lowerBound + (upperBound - lowerBound)*e/(double)energyResolution)/scaleFactor;
					greensFunctionData[e] += coefficients[n]*conj((1/scaleFactor)*(-2.*i/sqrt(1+DELTA - E*E))*exp(-i*((double)n)*acos(E))/denominator);
				}
			}
		}
	}
	else if(type == Type::Principal){
		if(generatingFunctionLookupTable){
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

				for(int e = 0; e < energyResolution; e++){
					double E = (lowerBound + (upperBound - lowerBound)*e/(double)energyResolution)/scaleFactor;
					greensFunctionData[e] += -coefficients[n]*real((1/scaleFactor)*(-2.*i/sqrt(1+DELTA - E*E))*exp(-i*((double)n)*acos(E))/denominator);
				}
			}
		}
	}
	else if(type == Type::NonPrincipal){
		if(generatingFunctionLookupTable){
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

				for(int e = 0; e < energyResolution; e++){
					double E = (lowerBound + (upperBound - lowerBound)*e/(double)energyResolution)/scaleFactor;
					greensFunctionData[e] -= coefficients[n]*i*imag((1/scaleFactor)*(-2.*i/sqrt(1+DELTA - E*E))*exp(-i*((double)n)*acos(E))/denominator);
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

/*complex<double>* ChebyshevExpander::generateGreensFunctionCPU(
	complex<double> *coefficients,
//	Property::GreensFunction::Type type
	Type type
){
	TBTKAssert(
		generatingFunctionLookupTable != NULL,
		"ChebyshevExpander::generateGreensFunction()",
		"Lookup table has not been generated.",
		"Use ChebyshevExpander::generateLookupTable() to generate lookup table."
	);

	complex<double> *greensFunctionData = new complex<double>[lookupTableResolution];
	for(int e = 0; e < lookupTableResolution; e++)
		greensFunctionData[e] = 0.;

	if(type == Type::Retarded){
		for(int n = 0; n < lookupTableNumCoefficients; n++){
			for(int e = 0; e < lookupTableResolution; e++){
				greensFunctionData[e] += generatingFunctionLookupTable[n][e]*coefficients[n];
			}
		}
	}
	else if(type == Type::Advanced){
		for(int n = 0; n < lookupTableNumCoefficients; n++){
			for(int e = 0; e < lookupTableResolution; e++){
				greensFunctionData[e] += coefficients[n]*conj(generatingFunctionLookupTable[n][e]);
			}
		}
	}
	else if(type == Type::Principal){
		for(int n = 0; n < lookupTableNumCoefficients; n++){
			for(int e = 0; e < lookupTableResolution; e++){
				greensFunctionData[e] += -coefficients[n]*real(generatingFunctionLookupTable[n][e]);
			}
		}
	}
	else if(type == Type::NonPrincipal){
		for(int n = 0; n < lookupTableNumCoefficients; n++){
			for(int e = 0; e < lookupTableResolution; e++){
				greensFunctionData[e] -= coefficients[n]*i*imag(generatingFunctionLookupTable[n][e]);
			}
		}
	}

	return greensFunctionData;
}*/

complex<double> ChebyshevExpander::getMonolopoulosABCDamping(
	double distanceToBoundary,
	double boundarySize,
	double e,
	double c
){
	complex<double> gamma = 0.;

	if(distanceToBoundary < 0){
		return 0.;
	}
	else if(distanceToBoundary < boundarySize){
		double hbar = UnitHandler::getConstantInNaturalUnits("hbar");
		double m = UnitHandler::getConstantInNaturalUnits("m_e");
		double y = c*(boundarySize - distanceToBoundary)/boundarySize;
		double f = 4./pow(c-y, 2) + 4./pow(c+y, 2) - 8./pow(c, 2);
		gamma = asinh(e*(pow(hbar, 2)/(2*m))*pow(2*M_PI/boundarySize, 2)*(f/scaleFactor));
	}

	return exp(-gamma);
}

};	//End of namespace Solver
};	//End of namespace TBTK
