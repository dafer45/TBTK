/** @file PropertyExtractorChebyshev.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/PropertyExtractorChebyshev.h"

using namespace std;

PropertyExtractorChebyshev::PropertyExtractorChebyshev(ChebyshevSolver *cSolver,
							int numCoefficients,
							int energyResolution,
							bool useGPUToCalculateCoefficients,
							bool useGPUToGenerateGreensFunctions,
							bool useLookupTable){
	this->cSolver = cSolver;
	this->numCoefficients = numCoefficients;
	this->energyResolution = energyResolution;
	this->useGPUToCalculateCoefficients = useGPUToCalculateCoefficients;
	this->useGPUToGenerateGreensFunctions = useGPUToGenerateGreensFunctions;
	this->useLookupTable = useLookupTable;

	if(useLookupTable){
		cSolver->generateLookupTable(numCoefficients, energyResolution);
		if(useGPUToGenerateGreensFunctions)
			cSolver->loadLookupTableGPU();
	}
	else if(useGPUToGenerateGreensFunctions){
		cout << "Error in PropertyExtractorChebyshev: useLookupTable cannot be false if useGPUToGenerateGreensFunction is true.\n";
		exit(1);
	}
}

PropertyExtractorChebyshev::~PropertyExtractorChebyshev(){
	if(useGPUToGenerateGreensFunctions)
		cSolver->destroyLookupTableGPU();

//	if(useLookupTable)
}

complex<double>* PropertyExtractorChebyshev::calculateGreensFunction(Index to, Index from){
	vector<Index> toIndices;
	toIndices.push_back(to);

	return calculateGreensFunctions(toIndices, from);
}

complex<double>* PropertyExtractorChebyshev::calculateGreensFunctions(vector<Index> &to, Index from){
	complex<double> *coefficients = new complex<double>[energyResolution*to.size()];

	if(useGPUToCalculateCoefficients){
		cSolver->calculateCoefficientsGPU(to, from, coefficients, numCoefficients);
	}
	else{
		cout << "Error in PropertyExtractorChebyshev::calculateGreensFunctions: CPU generation of coefficients not yet supported.\n";
		exit(1);
	}

	complex<double> *greensFunction = new complex<double>[energyResolution*to.size()];

	if(useGPUToGenerateGreensFunctions){
		for(unsigned int n = 0; n < to.size(); n++){
			cSolver->generateGreensFunctionGPU(greensFunction,
								&(coefficients[n*numCoefficients]));
		}
	}
	else{
		if(useLookupTable){
			#pragma omp parallel for
			for(unsigned int n = 0; n < to.size(); n++){
				cSolver->generateGreensFunction(greensFunction,
								&(coefficients[n*numCoefficients]));
			}
		}
		else{
			#pragma omp parallel for
			for(unsigned int n = 0; n < to.size(); n++){
				cSolver->generateGreensFunction(greensFunction,
								&(coefficients[n*numCoefficients]),
								numCoefficients,
								energyResolution);
			}
		}
	}

	delete [] coefficients;

	return greensFunction;
}

double* PropertyExtractorChebyshev::calculateLDOS(Index pattern, Index ranges){
	//To be implemented
	return NULL;
}
