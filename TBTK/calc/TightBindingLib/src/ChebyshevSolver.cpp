/** @file ChebyshevSolver.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/ChebyshevSolver.h"
#include <math.h>
#include "../include/HALinkedList.h"

using namespace std;

const complex<double> i(0, 1);

ChebyshevSolver::ChebyshevSolver(){
	generatingFunctionLookupTable = NULL;
	generatingFunctionLookupTable_device = NULL;
	lookupTableNumCoefficients = 0;
	lookupTableResolution = 0;
}

ChebyshevSolver::~ChebyshevSolver(){
	if(generatingFunctionLookupTable != NULL){
		for(int n = 0; n < lookupTableNumCoefficients; n++)
			delete [] generatingFunctionLookupTable[n];

		delete [] generatingFunctionLookupTable;
	}
}

void ChebyshevSolver::setModel(Model *model){
	this->model = model;
}

void ChebyshevSolver::calculateCoefficients(Index to, Index from, complex<double> *coefficients, int numCoefficients, double broadening){
	AmplitudeSet *amplitudeSet = &model->amplitudeSet;

	int fromBasisIndex = amplitudeSet->getBasisIndex(from);
	int toBasisIndex = amplitudeSet->getBasisIndex(to);

	cout << "ChebyshevSolver::calculateCoefficients\n";
	cout << "\tFrom Index: " << fromBasisIndex << "\n";
	cout << "\tTo Index: " << toBasisIndex << "\n";
	cout << "\tBasis size: " << amplitudeSet->getBasisSize() << "\n";
	cout << "\tProgress (100 coefficients per dot): ";

	complex<double> *jIn1 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jIn2 = new complex<double>[amplitudeSet->getBasisSize()];
//	complex<double> *jOut = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jResult = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
//		jOut[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;
//	jOut[toBasisIndex] = 1.;

	coefficients[0] = jIn1[toBasisIndex];

	//Generate a fixed hopping amplitude and inde list, for speed.
	AmplitudeSet::iterator it = amplitudeSet->getIterator();
	HoppingAmplitude *ha;
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
		toIndices[counter] = amplitudeSet->getBasisIndex(ha->toIndex);
		fromIndices[counter] = amplitudeSet->getBasisIndex(ha->fromIndex);
		hoppingAmplitudes[counter] = ha->getAmplitude();

		it.searchNextHA();
		counter++;
	}

	//Calculate |j1>
	for(int c = 0; c < amplitudeSet->getBasisSize(); c++)
		jResult[c] = 0.;
	for(int n = 0; n < numHoppingAmplitudes; n++){
		int from = fromIndices[n];
		int to = toIndices[n];

		jResult[to] += hoppingAmplitudes[n]*jIn1[from];
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
		for(int c = 0; c < amplitudeSet->getBasisSize(); c++)
			jResult[c] = -jIn2[c];
		for(int c = 0; c < numHoppingAmplitudes; c++){
			int from = fromIndices[c];
			int to = toIndices[c];

			jResult[to] += hoppingAmplitudes[c]*jIn1[from];
		}

		jTemp = jIn2;
		jIn2 = jIn1;
		jIn1 = jResult;
		jResult = jTemp;

		coefficients[n] = jIn1[toBasisIndex];
		if(n%100 == 0)
			cout << ".";
		if(n%1000 == 0)
			cout << " ";
	}

	delete [] jIn1;
	delete [] jIn2;
//	delete [] jOut;
	delete [] jResult;
	delete [] hoppingAmplitudes;
	delete [] toIndices;
	delete [] fromIndices;

	//Lorentzian convolution
//	double epsilon = 0.001;
//	double lambda = epsilon*numCoefficients;
	double lambda = broadening*numCoefficients;
	for(int n = 0; n < numCoefficients; n++)
		coefficients[n] = coefficients[n]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
}

void ChebyshevSolver::calculateCoefficientsWithCutoff(Index to, Index from, complex<double> *coefficients, int numCoefficients, double componentCutoff, double broadening){
	AmplitudeSet *amplitudeSet = &model->amplitudeSet;

	int fromBasisIndex = amplitudeSet->getBasisIndex(from);
	int toBasisIndex = amplitudeSet->getBasisIndex(to);

	cout << "ChebyshevSolver::calculateCoefficients\n";
	cout << "\tFrom Index: " << fromBasisIndex << "\n";
	cout << "\tTo Index: " << toBasisIndex << "\n";
	cout << "\tBasis size: " << amplitudeSet->getBasisSize() << "\n";
	cout << "\tProgress (100 coefficients per dot): ";

	complex<double> *jIn1 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jIn2 = new complex<double>[amplitudeSet->getBasisSize()];
//	complex<double> *jOut = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jResult = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
//		jOut[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state |j0>.
	jIn1[fromBasisIndex] = 1.;
//	jOut[toBasisIndex] = 1.;

	coefficients[0] = jIn1[toBasisIndex];

	HALinkedList haLinkedList(*amplitudeSet);
	haLinkedList.addLinkedList(fromBasisIndex);

	int *newlyReachedIndices = new int[amplitudeSet->getBasisSize()];
	int *everReachedIndices = new int[amplitudeSet->getBasisSize()];
	bool *everReachedIndicesAdded = new bool[amplitudeSet->getBasisSize()];
	int newlyReachedIndicesCounter = 0;
	int everReachedIndicesCounter = 0;
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
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
/*		if(n%100 == 0)
			cout << n << "\t" << everReachedIndicesCounter
				<< "\t" << everReachedIndicesCounter/(double)amplitudeSet->getBasisSize()
				<< "\t" << abs(coefficients[n])
//				<< "\t" << sqrt(abs(scalarProduct(jIn1, jIn1, amplitudeSet->getBasisSize())))
				<< "\n";*/
		if(n%100 == 0)
			cout << ".";
		if(n%1000 == 0)
			cout << " ";
	}

	delete [] jIn1;
	delete [] jIn2;
//	delete [] jOut;
	delete [] jResult;
	delete [] newlyReachedIndices;
	delete [] everReachedIndices;

	//Lorentzian convolution
//	double epsilon = 0.001;
//	double lambda = epsilon*numCoefficients;
	double lambda = broadening*numCoefficients;
	for(int n = 0; n < numCoefficients; n++)
		coefficients[n] = coefficients[n]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
}

void ChebyshevSolver::generateLookupTable(int numCoefficients, int energyResolution){
	cout << "Generating lookup table\n";
	cout << "\tNum coefficients: " << numCoefficients << "\n";
	cout << "\tEnergy resolution: " << energyResolution << "\n";

	if(generatingFunctionLookupTable != NULL){
		for(int n = 0; n < lookupTableNumCoefficients; n++)
			delete [] generatingFunctionLookupTable[n];

		delete [] generatingFunctionLookupTable;
	}

	lookupTableNumCoefficients = numCoefficients;
	lookupTableResolution = energyResolution;

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
			double E = -1 + 2.*e/(double)energyResolution;
			generatingFunctionLookupTable[n][e] = (-2.*i/sqrt(1+DELTA - E*E))*exp(-i*((double)n)*acos(E))/denominator;
		}
	}
}

void ChebyshevSolver::generateGreensFunction(complex<double> *greensFunction, complex<double> *coefficients, int numCoefficients, int energyResolution){
	for(int e = 0; e < energyResolution; e++)
		greensFunction[e] = 0.;

	const double DELTA = 0.0001;
	for(int n = 0; n < numCoefficients; n++){
		double denominator = 1.;
		if(n == 0)
			denominator = 2.;

		for(int e = 0; e < energyResolution; e++){
			double E = -1 + 2.*e/(double)energyResolution;
			greensFunction[e] += (-2.*i/sqrt(1+DELTA - E*E))*coefficients[n]*exp(-i*((double)n)*acos(E))/denominator;
		}
	}
}

void ChebyshevSolver::generateGreensFunction(complex<double> *greensFunction, complex<double> *coefficients){
	if(generatingFunctionLookupTable == NULL){
		cout << "Error in ChebyshevSolver::generateGreensFunction: Lookup table has not been generated.";
		exit(1);
	}
	else{
		for(int e = 0; e < lookupTableResolution; e++)
			greensFunction[e] = 0.;

		for(int n = 0; n < lookupTableNumCoefficients; n++){
			for(int e = 0; e < lookupTableResolution; e++){
				greensFunction[e] += generatingFunctionLookupTable[n][e]*coefficients[n];
			}
		}
	}
}
