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

/** @file ChebyshevSolver.cpp
 *
 *  @author Kristofer Björnson
 */

#include "ChebyshevSolver.h"
#include "HALinkedList.h"
#include "Streams.h"
#include "TBTKMacros.h"
#include "UnitHandler.h"

#include <iostream>
#include <math.h>

using namespace std;

namespace TBTK{

namespace{
	const complex<double> i(0, 1);
}

ChebyshevSolver::ChebyshevSolver(){
	scaleFactor = 1.;
	damping = NULL;
	generatingFunctionLookupTable = NULL;
	generatingFunctionLookupTable_device = NULL;
	lookupTableNumCoefficients = 0;
	lookupTableResolution = 0;
	lookupTableLowerBound = 0.;
	lookupTableUpperBound = 0.;
	isTalkative = false;
}

ChebyshevSolver::~ChebyshevSolver(){
	if(generatingFunctionLookupTable != NULL){
		for(int n = 0; n < lookupTableNumCoefficients; n++)
			delete [] generatingFunctionLookupTable[n];

		delete [] generatingFunctionLookupTable;
	}
}

void ChebyshevSolver::setModel(Model *model){
	Solver::setModel(model);
	model->sortHoppingAmplitudes();	//Required for GPU evaluation
}

void ChebyshevSolver::calculateCoefficients(
	Index to,
	Index from,
	complex<double> *coefficients,
	int numCoefficients,
	double broadening
){
	Model *model = getModel();

	TBTKAssert(
		model != NULL,
		"ChebyshevSolver::calculateCoefficients()",
		"Model not set.",
		"Use ChebyshevSolver::setModel() to set model."
	);
	TBTKAssert(
		scaleFactor > 0,
		"ChebyshevSolver::calculateCoefficients()",
		"Scale factor must be larger than zero.",
		"Use ChebyshevSolver::setScaleFactor() to set scale factor."
	);
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevSolver::calculateCoefficients()",
		"numCoefficients has to be larger than 0.",
		""
	);

	const HoppingAmplitudeSet *hoppingAmplitudeSet = model->getHoppingAmplitudeSet();

	int fromBasisIndex = hoppingAmplitudeSet->getBasisIndex(from);
	int toBasisIndex = hoppingAmplitudeSet->getBasisIndex(to);

	if(isTalkative){
		Streams::out << "ChebyshevSolver::calculateCoefficients\n";
		Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Streams::out << "\tTo Index: " << toBasisIndex << "\n";
		Streams::out << "\tBasis size: " << hoppingAmplitudeSet->getBasisSize() << "\n";
		Streams::out << "\tProgress (100 coefficients per dot): ";
	}

	complex<double> *jIn1 = new complex<double>[hoppingAmplitudeSet->getBasisSize()];
	complex<double> *jIn2 = new complex<double>[hoppingAmplitudeSet->getBasisSize()];
	complex<double> *jResult = new complex<double>[hoppingAmplitudeSet->getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	coefficients[0] = jIn1[toBasisIndex];

	//Generate a fixed hopping amplitude and inde list, for speed.
	HoppingAmplitudeSet::Iterator it = hoppingAmplitudeSet->getIterator();
	const HoppingAmplitude *ha;
	int numHoppingAmplitudes = 0;
	while((ha = it.getHA())){
		numHoppingAmplitudes++;
		it.searchNextHA();
	}

	complex<double> *hoppingAmplitudes = new complex<double>[numHoppingAmplitudes];
	int *toIndices = new int[numHoppingAmplitudes];
	int *fromIndices = new int[numHoppingAmplitudes];
	it.reset();
	int counter = 0;
	while((ha = it.getHA())){
		toIndices[counter] = hoppingAmplitudeSet->getBasisIndex(ha->toIndex);
		fromIndices[counter] = hoppingAmplitudeSet->getBasisIndex(ha->fromIndex);
		hoppingAmplitudes[counter] = ha->getAmplitude()/scaleFactor;

		it.searchNextHA();
		counter++;
	}

	//Calculate |j1>
	for(int c = 0; c < hoppingAmplitudeSet->getBasisSize(); c++)
		jResult[c] = 0.;
	for(int n = 0; n < numHoppingAmplitudes; n++){
		int from = fromIndices[n];
		int to = toIndices[n];

		jResult[to] += hoppingAmplitudes[n]*jIn1[from];
	}

	if(damping != NULL){
		for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++)
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
		for(int c = 0; c < hoppingAmplitudeSet->getBasisSize(); c++)
			jResult[c] = -jIn2[c];

		if(damping != NULL){
			for(int c = 0; c < hoppingAmplitudeSet->getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		for(int c = 0; c < numHoppingAmplitudes; c++){
			int from = fromIndices[c];
			int to = toIndices[c];

			jResult[to] += hoppingAmplitudes[c]*jIn1[from];
		}

		if(damping != NULL){
			for(int c = 0; c < hoppingAmplitudeSet->getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		jTemp = jIn2;
		jIn2 = jIn1;
		jIn1 = jResult;
		jResult = jTemp;

		coefficients[n] = jIn1[toBasisIndex];

		if(isTalkative){
			if(n%100 == 0)
				Streams::out << "." << flush;
			if(n%1000 == 0)
				Streams::out << " " << flush;
		}
	}
	if(isTalkative)
		Streams::out << "\n";

	delete [] jIn1;
	delete [] jIn2;
	delete [] jResult;
	delete [] hoppingAmplitudes;
	delete [] toIndices;
	delete [] fromIndices;

	//Lorentzian convolution
	double lambda = broadening*numCoefficients;
	for(int n = 0; n < numCoefficients; n++)
		coefficients[n] = coefficients[n]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
}

void ChebyshevSolver::calculateCoefficients(
	vector<Index> &to,
	Index from,
	complex<double> *coefficients,
	int numCoefficients,
	double broadening
){
	Model *model = getModel();
	TBTKAssert(
		model != NULL,
		"ChebyshevSolver::calculateCoefficients()",
		"Model not set.",
		"Use ChebyshevSolver::setModel() to set model."
	);
	TBTKAssert(
		scaleFactor > 0,
		"ChebyshevSolver::calculateCoefficients()",
		"Scale factor must be larger than zero.",
		"Use ChebyshevSolver::setScaleFactor() to set scale factor."
	);
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevSolver::calculateCoefficients()",
		"numCoefficients has to be larger than 0.",
		""
	);

	const HoppingAmplitudeSet *hoppingAmplitudeSet = model->getHoppingAmplitudeSet();

	int fromBasisIndex = hoppingAmplitudeSet->getBasisIndex(from);
	int *coefficientMap = new int[hoppingAmplitudeSet->getBasisSize()];
	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++)
		coefficientMap[n] = -1;
	for(unsigned int n = 0; n < to.size(); n++)
		coefficientMap[hoppingAmplitudeSet->getBasisIndex(to.at(n))] = n;

	if(isTalkative){
		Streams::out << "ChebyshevSolver::calculateCoefficients\n";
		Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Streams::out << "\tBasis size: " << hoppingAmplitudeSet->getBasisSize() << "\n";
		Streams::out << "\tProgress (100 coefficients per dot): ";
	}

	complex<double> *jIn1 = new complex<double>[hoppingAmplitudeSet->getBasisSize()];
	complex<double> *jIn2 = new complex<double>[hoppingAmplitudeSet->getBasisSize()];
	complex<double> *jResult = new complex<double>[hoppingAmplitudeSet->getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]*numCoefficients] = jIn1[n];

	//Generate a fixed hopping amplitude and inde list, for speed.
	HoppingAmplitudeSet::Iterator it = hoppingAmplitudeSet->getIterator();
	const HoppingAmplitude *ha;
	int numHoppingAmplitudes = 0;
	while((ha = it.getHA())){
		numHoppingAmplitudes++;
		it.searchNextHA();
	}

	complex<double> *hoppingAmplitudes = new complex<double>[numHoppingAmplitudes];
	int *toIndices = new int[numHoppingAmplitudes];
	int *fromIndices = new int[numHoppingAmplitudes];
	it.reset();
	int counter = 0;
	while((ha = it.getHA())){
		toIndices[counter] = hoppingAmplitudeSet->getBasisIndex(ha->toIndex);
		fromIndices[counter] = hoppingAmplitudeSet->getBasisIndex(ha->fromIndex);
		hoppingAmplitudes[counter] = ha->getAmplitude()/scaleFactor;

		it.searchNextHA();
		counter++;
	}

	//Calculate |j1>
	for(int c = 0; c < hoppingAmplitudeSet->getBasisSize(); c++)
		jResult[c] = 0.;
	for(int n = 0; n < numHoppingAmplitudes; n++){
		int from = fromIndices[n];
		int to = toIndices[n];

		jResult[to] += hoppingAmplitudes[n]*jIn1[from];
	}

	if(damping != NULL){
		for(int c = 0; c < hoppingAmplitudeSet->getBasisSize(); c++)
			jResult[c] *= damping[c];
	}

	jTemp = jIn2;
	jIn2 = jIn1;
	jIn1 = jResult;
	jResult = jTemp;

	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]*numCoefficients + 1] = jIn1[n];

	//Multiply hopping amplitudes by factor two, to spped up calculation of 2H|j(n-1)> - |j(n-2)>.
	for(int n = 0; n < numHoppingAmplitudes; n++)
		hoppingAmplitudes[n] *= 2.;

	//Iteratively calculate |jn> and corresponding Chebyshev coefficients.
	for(int n = 2; n < numCoefficients; n++){
		for(int c = 0; c < hoppingAmplitudeSet->getBasisSize(); c++)
			jResult[c] = -jIn2[c];

		if(damping != NULL){
			for(int c = 0; c < hoppingAmplitudeSet->getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		for(int c = 0; c < numHoppingAmplitudes; c++){
			int from = fromIndices[c];
			int to = toIndices[c];

			jResult[to] += hoppingAmplitudes[c]*jIn1[from];
		}

		if(damping != NULL){
			for(int c = 0; c < hoppingAmplitudeSet->getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		jTemp = jIn2;
		jIn2 = jIn1;
		jIn1 = jResult;
		jResult = jTemp;

		for(int c = 0; c < hoppingAmplitudeSet->getBasisSize(); c++)
			if(coefficientMap[c] != -1)
				coefficients[coefficientMap[c]*numCoefficients + n] = jIn1[c];

		if(isTalkative){
			if(n%100 == 0)
				Streams::out << "." << flush;
			if(n%1000 == 0)
				Streams::out << " " << flush;
		}
	}
	if(isTalkative)
		Streams::out << "\n";

	delete [] jIn1;
	delete [] jIn2;
	delete [] jResult;
	delete [] hoppingAmplitudes;
	delete [] toIndices;
	delete [] fromIndices;

	//Lorentzian convolution
	double lambda = broadening*numCoefficients;
	for(int n = 0; n < numCoefficients; n++)
		coefficients[n] = coefficients[n]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
}

void ChebyshevSolver::calculateCoefficientsWithCutoff(
	Index to,
	Index from,
	complex<double> *coefficients,
	int numCoefficients,
	double componentCutoff,
	double broadening
){
	Model *model = getModel();

	TBTKAssert(
		model != NULL,
		"ChebyshevSolver::calculateCoefficientsWithCutoff()",
		"Model not set.",
		"Use ChebyshevSolver::setModel() to set model."
	);
	TBTKAssert(
		scaleFactor > 0,
		"ChebyshevSolver::calculateCoefficientsWithCutoff()",
		"Scale factor must be larger than zero.",
		"Use ChebyshevSolver::setScaleFactor() to set scale factor."
	);
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevSolver::calculateCoefficientsWithCutoff()",
		"numCoefficients has to be larger than 0.",
		""
	);

	const HoppingAmplitudeSet *hoppingAmplitudeSet = model->getHoppingAmplitudeSet();

	int fromBasisIndex = hoppingAmplitudeSet->getBasisIndex(from);
	int toBasisIndex = hoppingAmplitudeSet->getBasisIndex(to);

	if(isTalkative){
		Streams::out << "ChebyshevSolver::calculateCoefficients\n";
		Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Streams::out << "\tTo Index: " << toBasisIndex << "\n";
		Streams::out << "\tBasis size: " << hoppingAmplitudeSet->getBasisSize() << "\n";
		Streams::out << "\tProgress (100 coefficients per dot): ";
	}

	complex<double> *jIn1 = new complex<double>[hoppingAmplitudeSet->getBasisSize()];
	complex<double> *jIn2 = new complex<double>[hoppingAmplitudeSet->getBasisSize()];
	complex<double> *jResult = new complex<double>[hoppingAmplitudeSet->getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state |j0>.
	jIn1[fromBasisIndex] = 1.;

	coefficients[0] = jIn1[toBasisIndex];

	HALinkedList haLinkedList(*hoppingAmplitudeSet);
	haLinkedList.rescaleAmplitudes(scaleFactor);
	haLinkedList.addLinkedList(fromBasisIndex);

	int *newlyReachedIndices = new int[hoppingAmplitudeSet->getBasisSize()];
	int *everReachedIndices = new int[hoppingAmplitudeSet->getBasisSize()];
	bool *everReachedIndicesAdded = new bool[hoppingAmplitudeSet->getBasisSize()];
	int newlyReachedIndicesCounter = 0;
	int everReachedIndicesCounter = 0;
	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++){
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

		if(isTalkative){
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
	double lambda = broadening*numCoefficients;
	for(int n = 0; n < numCoefficients; n++)
		coefficients[n] = coefficients[n]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
}

void ChebyshevSolver::generateLookupTable(
	int numCoefficients,
	int energyResolution,
	double lowerBound,
	double upperBound
){
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevSolver::generateLookupTable()",
		"numCoefficients has to be larger than 0.",
		""
	);
	TBTKAssert(
		energyResolution > 0,
		"ChebyshevSolver::generateLookupTable()",
		"energyResolution has to be larger than 0.",
		""
	);
	TBTKAssert(
		lowerBound < upperBound,
		"ChebyshevSolver::generateLookupTable()",
		"lowerBound has to be smaller than upperBound.",
		""
	);
	TBTKAssert(
		lowerBound >= -scaleFactor,
		"ChebyshevSolver::generateLookupTable()",
		"lowerBound has to be larger than -scaleFactor.",
		"Use ChebyshevSolver::setScaleFactor to set a larger scale factor."
	);
	TBTKAssert(
		upperBound <= scaleFactor,
		"ChebyshevSolver::generateLookupTable()",
		"upperBound has to be smaller than scaleFactor.",
		"Use ChebyshevSolver::setScaleFactor to set a larger scale factor."
	);

	if(isTalkative){
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

void ChebyshevSolver::destroyLookupTable(){
	TBTKAssert(
		generatingFunctionLookupTable != NULL,
		"ChebyshevSolver::destroyLookupTable()",
		"No lookup table generated.",
		""
	);
	for(int n = 0; n < lookupTableNumCoefficients; n++)
		delete [] generatingFunctionLookupTable[n];

	delete [] generatingFunctionLookupTable;

	generatingFunctionLookupTable = NULL;
}

Property::GreensFunction* ChebyshevSolver::generateGreensFunction(
	complex<double> *coefficients,
	int numCoefficients,
	int energyResolution,
	double lowerBound,
	double upperBound,
	Property::GreensFunction::Type type
){
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevSolver::generateLookupTable()",
		"numCoefficients has to be larger than 0.",
		""
	);
	TBTKAssert(
		energyResolution > 0,
		"ChebyshevSolver::generateLookupTable()",
		"energyResolution has to be larger than 0.",
		""
	);
	TBTKAssert(
		lowerBound < upperBound,
		"ChebyshevSolver::generateLookupTable()",
		"lowerBound has to be smaller than upperBound.",
		""
	);
	TBTKAssert(
		lowerBound > -scaleFactor,
		"ChebyshevSolver::generateLookupTable()",
		"lowerBound has to be larger than -scaleFactor.",
		"Use ChebyshevSolver::setScaleFactor to set a larger scale factor."
	);
	TBTKAssert(
		upperBound < scaleFactor,
		"ChebyshevSolver::generateLookupTable()",
		"upperBound has to be smaller than scaleFactor.",
		"Use ChebyshevSolver::setScaleFactor to set a larger scale factor."
	);

	complex<double> *greensFunctionData = new complex<double>[energyResolution];
	for(int e = 0; e < energyResolution; e++)
		greensFunctionData[e] = 0.;

	const double DELTA = 0.0001;
	if(type == Property::GreensFunction::Type::Retarded){
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
	else if(type == Property::GreensFunction::Type::Advanced){
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
	else if(type == Property::GreensFunction::Type::Principal){
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
	else if(type == Property::GreensFunction::Type::NonPrincipal){
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
	else{
		TBTKExit(
			"ChebyshevSolver::generateGreensFunction()",
			"Unknown GreensFunctionType",
			""
		);
	}

	Property::GreensFunction *greensFunction = new Property::GreensFunction(
		type,
		Property::GreensFunction::Format::Array,
		lowerBound,
		upperBound,
		energyResolution,
		greensFunctionData
	);
	delete [] greensFunctionData;

	return greensFunction;
}

Property::GreensFunction* ChebyshevSolver::generateGreensFunction(
	complex<double> *coefficients,
	Property::GreensFunction::Type type
){
	TBTKAssert(
		generatingFunctionLookupTable != NULL,
		"ChebyshevSolver::generateGreensFunction()",
		"Lookup table has not been generated.",
		"Use ChebyshevSolver::generateLookupTable() to generate lookup table."
	);

	complex<double> *greensFunctionData = new complex<double>[lookupTableResolution];
	for(int e = 0; e < lookupTableResolution; e++)
		greensFunctionData[e] = 0.;

	if(type == Property::GreensFunction::Type::Retarded){
		for(int n = 0; n < lookupTableNumCoefficients; n++){
			for(int e = 0; e < lookupTableResolution; e++){
				greensFunctionData[e] += generatingFunctionLookupTable[n][e]*coefficients[n];
			}
		}
	}
	else if(type == Property::GreensFunction::Type::Advanced){
		for(int n = 0; n < lookupTableNumCoefficients; n++){
			for(int e = 0; e < lookupTableResolution; e++){
				greensFunctionData[e] += coefficients[n]*conj(generatingFunctionLookupTable[n][e]);
			}
		}
	}
	else if(type == Property::GreensFunction::Type::Principal){
		for(int n = 0; n < lookupTableNumCoefficients; n++){
			for(int e = 0; e < lookupTableResolution; e++){
				greensFunctionData[e] += -coefficients[n]*real(generatingFunctionLookupTable[n][e]);
			}
		}
	}
	else if(type == Property::GreensFunction::Type::NonPrincipal){
		for(int n = 0; n < lookupTableNumCoefficients; n++){
			for(int e = 0; e < lookupTableResolution; e++){
				greensFunctionData[e] -= coefficients[n]*i*imag(generatingFunctionLookupTable[n][e]);
			}
		}
	}

	Property::GreensFunction *greensFunction = new Property::GreensFunction(
		type,
		Property::GreensFunction::Format::Array,
		lookupTableLowerBound,
		lookupTableUpperBound,
		lookupTableResolution,
		greensFunctionData
	);
	delete [] greensFunctionData;

	return greensFunction;
}

complex<double> ChebyshevSolver::getMonolopoulosABCDamping(
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
		double hbar = UnitHandler::getHbarN();
		double m = UnitHandler::getM_eN();
		double y = c*(boundarySize - distanceToBoundary)/boundarySize;
		double f = 4./pow(c-y, 2) + 4./pow(c+y, 2) - 8./pow(c, 2);
		gamma = asinh(e*(pow(hbar, 2)/(2*m))*pow(2*M_PI/boundarySize, 2)*(f/scaleFactor));
	}

	return exp(-gamma);
}

};	//End of namespace TBTK
