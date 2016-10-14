/** @file ChebyshevSolver.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/ChebyshevSolver.h"
#include "../include/HALinkedList.h"
#include "../include/UnitHandler.h"
#include "../include/TBTKMacros.h"

#include <math.h>

using namespace std;

namespace TBTK{

const complex<double> i(0, 1);

int ChebyshevSolver::numChebyshevSolvers = 0;
int ChebyshevSolver::numDevices = 0;
bool *ChebyshevSolver::busyDevices = NULL;
omp_lock_t ChebyshevSolver::busyDevicesLock;
ChebyshevSolver::StaticConstructor ChebyshevSolver::staticConstructor;

ChebyshevSolver::StaticConstructor::StaticConstructor(){
	omp_init_lock(&ChebyshevSolver::busyDevicesLock);
}

ChebyshevSolver::ChebyshevSolver(){
	model = NULL;
	scaleFactor = 1.;
	damping = NULL;
	generatingFunctionLookupTable = NULL;
	generatingFunctionLookupTable_device = NULL;
	lookupTableNumCoefficients = 0;
	lookupTableResolution = 0;
	isTalkative = false;

	omp_set_lock(&busyDevicesLock);
	#pragma omp flush
	{
		if(numChebyshevSolvers == 0)
			createDeviceTableGPU();
		numChebyshevSolvers++;
	}
	#pragma omp flush
	omp_unset_lock(&busyDevicesLock);
}

ChebyshevSolver::~ChebyshevSolver(){
	if(generatingFunctionLookupTable != NULL){
		for(int n = 0; n < lookupTableNumCoefficients; n++)
			delete [] generatingFunctionLookupTable[n];

		delete [] generatingFunctionLookupTable;
	}

	omp_set_lock(&busyDevicesLock);
	#pragma omp flush
	{
		numChebyshevSolvers--;
		if(numChebyshevSolvers == 0)
			destroyDeviceTableGPU();
	}
	#pragma omp flush
	omp_unset_lock(&busyDevicesLock);
}

void ChebyshevSolver::setModel(Model *model){
	this->model = model;
	model->getAmplitudeSet()->sort();	//Required for GPU evaluation
}

void ChebyshevSolver::calculateCoefficients(
	Index to,
	Index from,
	complex<double> *coefficients,
	int numCoefficients,
	double broadening
){
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

	AmplitudeSet *amplitudeSet = model->getAmplitudeSet();

	int fromBasisIndex = amplitudeSet->getBasisIndex(from);
	int toBasisIndex = amplitudeSet->getBasisIndex(to);

	if(isTalkative){
		cout << "ChebyshevSolver::calculateCoefficients\n";
		cout << "\tFrom Index: " << fromBasisIndex << "\n";
		cout << "\tTo Index: " << toBasisIndex << "\n";
		cout << "\tBasis size: " << amplitudeSet->getBasisSize() << "\n";
		cout << "\tProgress (100 coefficients per dot): ";
	}

	complex<double> *jIn1 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jIn2 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jResult = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	coefficients[0] = jIn1[toBasisIndex];

	//Generate a fixed hopping amplitude and inde list, for speed.
	AmplitudeSet::Iterator it = amplitudeSet->getIterator();
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
		hoppingAmplitudes[counter] = ha->getAmplitude()/scaleFactor;

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

	if(damping != NULL){
		for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
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
		for(int c = 0; c < amplitudeSet->getBasisSize(); c++)
			jResult[c] = -jIn2[c];

		if(damping != NULL){
			for(int c = 0; c < amplitudeSet->getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		for(int c = 0; c < numHoppingAmplitudes; c++){
			int from = fromIndices[c];
			int to = toIndices[c];

			jResult[to] += hoppingAmplitudes[c]*jIn1[from];
		}

		if(damping != NULL){
			for(int c = 0; c < amplitudeSet->getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		jTemp = jIn2;
		jIn2 = jIn1;
		jIn1 = jResult;
		jResult = jTemp;

		coefficients[n] = jIn1[toBasisIndex];

		if(isTalkative){
			if(n%100 == 0)
				cout << "." << flush;
			if(n%1000 == 0)
				cout << " " << flush;
		}
	}
	if(isTalkative)
		cout << "\n";

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

	AmplitudeSet *amplitudeSet = model->getAmplitudeSet();

	int fromBasisIndex = amplitudeSet->getBasisIndex(from);
	int *coefficientMap = new int[amplitudeSet->getBasisSize()];
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		coefficientMap[n] = -1;
	for(unsigned int n = 0; n < to.size(); n++)
		coefficientMap[amplitudeSet->getBasisIndex(to.at(n))] = n;

	if(isTalkative){
		cout << "ChebyshevSolver::calculateCoefficients\n";
		cout << "\tFrom Index: " << fromBasisIndex << "\n";
		cout << "\tBasis size: " << amplitudeSet->getBasisSize() << "\n";
		cout << "\tProgress (100 coefficients per dot): ";
	}

	complex<double> *jIn1 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jIn2 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jResult = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]*numCoefficients] = jIn1[n];

	//Generate a fixed hopping amplitude and inde list, for speed.
	AmplitudeSet::Iterator it = amplitudeSet->getIterator();
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
		hoppingAmplitudes[counter] = ha->getAmplitude()/scaleFactor;

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

	if(damping != NULL){
		for(int c = 0; c < amplitudeSet->getBasisSize(); c++)
			jResult[c] *= damping[c];
	}

	jTemp = jIn2;
	jIn2 = jIn1;
	jIn1 = jResult;
	jResult = jTemp;

	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]*numCoefficients + 1] = jIn1[n];

	//Multiply hopping amplitudes by factor two, to spped up calculation of 2H|j(n-1)> - |j(n-2)>.
	for(int n = 0; n < numHoppingAmplitudes; n++)
		hoppingAmplitudes[n] *= 2.;

	//Iteratively calculate |jn> and corresponding Chebyshev coefficients.
	for(int n = 2; n < numCoefficients; n++){
		for(int c = 0; c < amplitudeSet->getBasisSize(); c++)
			jResult[c] = -jIn2[c];

		if(damping != NULL){
			for(int c = 0; c < amplitudeSet->getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		for(int c = 0; c < numHoppingAmplitudes; c++){
			int from = fromIndices[c];
			int to = toIndices[c];

			jResult[to] += hoppingAmplitudes[c]*jIn1[from];
		}

		if(damping != NULL){
			for(int c = 0; c < amplitudeSet->getBasisSize(); c++)
				jResult[c] *= damping[c];
		}

		jTemp = jIn2;
		jIn2 = jIn1;
		jIn1 = jResult;
		jResult = jTemp;

		for(int c = 0; c < amplitudeSet->getBasisSize(); c++)
			if(coefficientMap[c] != -1)
				coefficients[coefficientMap[c]*numCoefficients + n] = jIn1[c];

		if(isTalkative){
			if(n%100 == 0)
				cout << "." << flush;
			if(n%1000 == 0)
				cout << " " << flush;
		}
	}
	if(isTalkative)
		cout << "\n";

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

	AmplitudeSet *amplitudeSet = model->getAmplitudeSet();

	int fromBasisIndex = amplitudeSet->getBasisIndex(from);
	int toBasisIndex = amplitudeSet->getBasisIndex(to);

	if(isTalkative){
		cout << "ChebyshevSolver::calculateCoefficients\n";
		cout << "\tFrom Index: " << fromBasisIndex << "\n";
		cout << "\tTo Index: " << toBasisIndex << "\n";
		cout << "\tBasis size: " << amplitudeSet->getBasisSize() << "\n";
		cout << "\tProgress (100 coefficients per dot): ";
	}

	complex<double> *jIn1 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jIn2 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jResult = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}
	//Set up initial state |j0>.
	jIn1[fromBasisIndex] = 1.;

	coefficients[0] = jIn1[toBasisIndex];

	HALinkedList haLinkedList(*amplitudeSet);
	haLinkedList.rescaleAmplitudes(scaleFactor);
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

		if(isTalkative){
/*			if(n%100 == 0)
				cout << n << "\t" << everReachedIndicesCounter
					<< "\t" << everReachedIndicesCounter/(double)amplitudeSet->getBasisSize()
					<< "\t" << abs(coefficients[n])
//					<< "\t" << sqrt(abs(scalarProduct(jIn1, jIn1, amplitudeSet->getBasisSize())))
					<< "\n";*/
			if(n%100 == 0)
				cout << ".";
			if(n%1000 == 0)
				cout << " ";
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
	cout << lowerBound << " " << upperBound << "\n";
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
		cout << "Generating lookup table\n";
		cout << "\tNum coefficients: " << numCoefficients << "\n";
		cout << "\tEnergy resolution: " << energyResolution << "\n";
		cout << "\tLower bound: " << lowerBound << "\n";
		cout << "\tUpper bound: " << upperBound << "\n";
	}

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
			double E = (lowerBound + (upperBound - lowerBound)*e/(double)energyResolution)/scaleFactor;
			generatingFunctionLookupTable[n][e] = (1/scaleFactor)*(-2.*i/sqrt(1+DELTA - E*E))*exp(-i*((double)n)*acos(E))/denominator;
		}
	}
}

void ChebyshevSolver::generateGreensFunction(
	complex<double> *greensFunction,
	complex<double> *coefficients,
	int numCoefficients,
	int energyResolution,
	double lowerBound,
	double upperBound,
	GreensFunctionType type
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

	for(int e = 0; e < energyResolution; e++)
		greensFunction[e] = 0.;

	const double DELTA = 0.0001;
	if(type == GreensFunctionType::Retarded){
		for(int n = 0; n < numCoefficients; n++){
			double denominator = 1.;
			if(n == 0)
				denominator = 2.;

			for(int e = 0; e < energyResolution; e++){
				double E = (lowerBound + (upperBound - lowerBound)*e/(double)energyResolution)/scaleFactor;
				greensFunction[e] += coefficients[n]*(1/scaleFactor)*(-2.*i/sqrt(1+DELTA - E*E))*exp(-i*((double)n)*acos(E))/denominator;
			}
		}
	}
	else if(type == GreensFunctionType::Advanced){
		for(int n = 0; n < numCoefficients; n++){
			double denominator = 1.;
			if(n == 0)
				denominator = 2.;

			for(int e = 0; e < energyResolution; e++){
				double E = (lowerBound + (upperBound - lowerBound)*e/(double)energyResolution)/scaleFactor;
				greensFunction[e] += coefficients[n]*conj((1/scaleFactor)*(-2.*i/sqrt(1+DELTA - E*E))*exp(-i*((double)n)*acos(E))/denominator);
			}
		}
	}
	else if(type == GreensFunctionType::Principal){
		for(int n = 0; n < numCoefficients; n++){
			double denominator = 1.;
			if(n == 0)
				denominator = 2.;

			for(int e = 0; e < energyResolution; e++){
				double E = (lowerBound + (upperBound - lowerBound)*e/(double)energyResolution)/scaleFactor;
				greensFunction[e] += -coefficients[n]*real((1/scaleFactor)*(-2.*i/sqrt(1+DELTA - E*E))*exp(-i*((double)n)*acos(E))/denominator);
			}
		}
	}
	else if(type == GreensFunctionType::NonPrincipal){
		for(int n = 0; n < numCoefficients; n++){
			double denominator = 1.;
			if(n == 0)
				denominator = 2.;

			for(int e = 0; e < energyResolution; e++){
				double E = (lowerBound + (upperBound - lowerBound)*e/(double)energyResolution)/scaleFactor;
				greensFunction[e] -= coefficients[n]*i*imag((1/scaleFactor)*(-2.*i/sqrt(1+DELTA - E*E))*exp(-i*((double)n)*acos(E))/denominator);
			}
		}
	}
	else{
		cout << "Error in ChebyshevSolver::generateGreensFunction: Unknown GreensFunctionType\n";
		exit(1);
	}
}

void ChebyshevSolver::generateGreensFunction(
	complex<double> *greensFunction,
	complex<double> *coefficients,
	GreensFunctionType type
){
	if(generatingFunctionLookupTable == NULL){
		cout << "Error in ChebyshevSolver::generateGreensFunction: Lookup table has not been generated.";
		exit(1);
	}
	else{
		for(int e = 0; e < lookupTableResolution; e++)
			greensFunction[e] = 0.;

		if(type == GreensFunctionType::Retarded){
			for(int n = 0; n < lookupTableNumCoefficients; n++){
				for(int e = 0; e < lookupTableResolution; e++){
					greensFunction[e] += generatingFunctionLookupTable[n][e]*coefficients[n];
				}
			}
		}
		else if(type == GreensFunctionType::Advanced){
			for(int n = 0; n < lookupTableNumCoefficients; n++){
				for(int e = 0; e < lookupTableResolution; e++){
					greensFunction[e] += coefficients[n]*conj(generatingFunctionLookupTable[n][e]);
				}
			}
		}
		else if(type == GreensFunctionType::Principal){
			for(int n = 0; n < lookupTableNumCoefficients; n++){
				for(int e = 0; e < lookupTableResolution; e++){
					greensFunction[e] += -coefficients[n]*real(generatingFunctionLookupTable[n][e]);
				}
			}
		}
		else if(type == GreensFunctionType::NonPrincipal){
			for(int n = 0; n < lookupTableNumCoefficients; n++){
				for(int e = 0; e < lookupTableResolution; e++){
					greensFunction[e] -= coefficients[n]*i*imag(generatingFunctionLookupTable[n][e]);
				}
			}
		}
	}
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
