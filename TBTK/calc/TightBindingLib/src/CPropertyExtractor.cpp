/** @file CPropertyExtractor.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/CPropertyExtractor.h"
#include "../include/Functions.h"

using namespace std;

namespace TBTK{

CPropertyExtractor::CPropertyExtractor(ChebyshevSolver *cSolver,
					int numCoefficients,
					int energyResolution,
					bool useGPUToCalculateCoefficients,
					bool useGPUToGenerateGreensFunctions,
					bool useLookupTable,
					double lowerBound,
					double upperBound){
	this->cSolver = cSolver;
	this->numCoefficients = numCoefficients;
	this->energyResolution = energyResolution;
	this->useGPUToCalculateCoefficients = useGPUToCalculateCoefficients;
	this->useGPUToGenerateGreensFunctions = useGPUToGenerateGreensFunctions;
	this->useLookupTable = useLookupTable;

	if(useLookupTable){
		cSolver->generateLookupTable(numCoefficients, energyResolution, lowerBound, upperBound);
		if(useGPUToGenerateGreensFunctions)
			cSolver->loadLookupTableGPU();
	}
	else if(useGPUToGenerateGreensFunctions){
		cout << "Error in PropertyExtractorChebyshev: useLookupTable cannot be false if useGPUToGenerateGreensFunction is true.\n";
		exit(1);
	}
}

CPropertyExtractor::~CPropertyExtractor(){
	if(useGPUToGenerateGreensFunctions)
		cSolver->destroyLookupTableGPU();
}

complex<double>* CPropertyExtractor::calculateGreensFunction(Index to, Index from){
	vector<Index> toIndices;
	toIndices.push_back(to);

	return calculateGreensFunctions(toIndices, from);
}

complex<double>* CPropertyExtractor::calculateGreensFunctions(vector<Index> &to, Index from){
	complex<double> *coefficients = new complex<double>[numCoefficients*to.size()];

	if(useGPUToCalculateCoefficients){
		cSolver->calculateCoefficientsGPU(to, from, coefficients, numCoefficients);
	}
	else{
		cSolver->calculateCoefficients(to, from, coefficients, numCoefficients);
//		cout << "Error in PropertyExtractorChebyshev::calculateGreensFunctions: CPU generation of coefficients not yet supported.\n";
//		exit(1);
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

double* CPropertyExtractor::calculateDensity(Index pattern, Index ranges){
	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) >= 0)
			ranges.indices.at(n) = 1;
	}

	int densityArraySize = 1;
	for(unsigned int n = 0; n < ranges.indices.size(); n++){
		if(pattern.indices.at(n) < IDX_SUM_ALL)
			densityArraySize *= ranges.indices.at(n);
	}
	double *density = new double[densityArraySize];
	for(int n = 0; n < densityArraySize; n++)
		density[n] = 0.;

	calculate(calculateDensityCallback, (void*)density, pattern, ranges, 0, 1);

	return density;
}

complex<double>* CPropertyExtractor::calculateMAG(Index pattern, Index ranges){
	hint = new int[1];
	((int*)hint)[0] = -1;
	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) == IDX_SPIN){
			((int*)hint)[0] = n;
			pattern.indices.at(n) = 0;
			ranges.indices.at(n) = 1;
			break;
		}
	}
	if(((int*)hint)[0] == -1){
		cout << "Error in PropertyExtractorChebyshev::calculateMAG: No spin index indicated.\n";
		delete [] (int*)hint;
		return NULL;
	}

	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) >= 0)
			ranges.indices.at(n) = 1;
	}

	int magArraySize = 1;
	for(unsigned int n = 0; n < ranges.indices.size(); n++){
		if(pattern.indices.at(n) < IDX_SUM_ALL)
			magArraySize *= ranges.indices.at(n);
	}
	complex<double> *mag = new complex<double>[4*magArraySize];
	for(int n = 0; n < 4*magArraySize; n++)
		mag[n] = 0.;

	calculate(calculateMAGCallback, (void*)mag, pattern, ranges, 0, 1);

	delete [] (int*)hint;

	return mag;
}

double* CPropertyExtractor::calculateLDOS(Index pattern, Index ranges){
	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) >= 0)
			ranges.indices.at(n) = 1;
	}

	int ldosArraySize = 1.;
	for(unsigned int n = 0; n < ranges.indices.size(); n++){
		if(pattern.indices.at(n) < IDX_SUM_ALL)
			ldosArraySize *= ranges.indices.at(n);
	}
	double *ldos = new double[energyResolution*ldosArraySize];
	for(int n = 0; n < energyResolution*ldosArraySize; n++)
		ldos[n] = 0.;

	calculate(calculateLDOSCallback, (void*)ldos, pattern, ranges, 0, 1);

	return ldos;
}

complex<double>* CPropertyExtractor::calculateSP_LDOS(Index pattern, Index ranges){
	hint = new int[1];
	((int*)hint)[0] = -1;
	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) == IDX_SPIN){
			((int*)hint)[0] = n;
			pattern.indices.at(n) = 0;
			ranges.indices.at(n) = 1;
			break;
		}
	}
	if(((int*)hint)[0] == -1){
		cout << "Error in PropertyExtractorChebyshev::calculateSP_LDOS: No spin index indicated.\n";
		delete [] (int*)hint;
		return NULL;
	}

	for(unsigned int n = 0; n < pattern.indices.size(); n++){
		if(pattern.indices.at(n) >= 0)
			ranges.indices.at(n) = 1;
	}

	int sp_ldosArraySize = 1;
	for(unsigned int n = 0; n < ranges.indices.size(); n++){
		if(pattern.indices.at(n) < IDX_SUM_ALL)
			sp_ldosArraySize *= ranges.indices.at(n);
	}
	complex<double> *sp_ldos = new complex<double>[4*energyResolution*sp_ldosArraySize];
	for(int n = 0; n < 4*energyResolution*sp_ldosArraySize; n++)
		sp_ldos[n] = 0.;

	calculate(calculateSP_LDOSCallback, (void*)sp_ldos, pattern, ranges, 0, 1);

	delete [] (int*)hint;

	return sp_ldos;
}

void CPropertyExtractor::calculateDensityCallback(CPropertyExtractor *cb_this, void *density, const Index &index, int offset){
	complex<double> *greensFunction = cb_this->calculateGreensFunction(index, index);

	for(int e = 0; e < cb_this->energyResolution; e++){
		double weight = Functions::fermiDiracDistribution((2.*e/(double)cb_this->energyResolution - 1.)*cb_this->cSolver->getScaleFactor(),
									cb_this->cSolver->getModel()->getFermiLevel(),
									cb_this->cSolver->getModel()->getTemperature());

		((double*)density)[offset] -= weight*imag(greensFunction[e])/M_PI;
	}

	delete [] greensFunction;
}

void CPropertyExtractor::calculateMAGCallback(CPropertyExtractor *cb_this, void *mag, const Index &index, int offset){
	int spinIndex = ((int*)(cb_this->hint))[0];
	Index to(index);
	Index from(index);
	complex<double> *greensFunction;

	for(int n = 0; n < 4; n++){
		to.indices.at(spinIndex) = n/2;		//up, up, down, down
		from.indices.at(spinIndex) = n%2;	//up, down, up, down
		greensFunction = cb_this->calculateGreensFunction(to, from);

		for(int e = 0; e < cb_this->energyResolution; e++){
			double weight = Functions::fermiDiracDistribution((2.*e/(double)cb_this->energyResolution - 1.)*cb_this->cSolver->getScaleFactor(),
										cb_this->cSolver->getModel()->getFermiLevel(),
										cb_this->cSolver->getModel()->getTemperature());

			((complex<double>*)mag)[4*offset + n] += weight*greensFunction[e];
		}

		delete [] greensFunction;
	}
}

void CPropertyExtractor::calculateLDOSCallback(CPropertyExtractor *cb_this, void *ldos, const Index &index, int offset){
	complex<double> *greensFunction = cb_this->calculateGreensFunction(index, index);

	for(int n = 0; n < cb_this->energyResolution; n++)
		((double*)ldos)[cb_this->energyResolution*offset + n] -= imag(greensFunction[n])/M_PI;

	delete [] greensFunction;
}

void CPropertyExtractor::calculateSP_LDOSCallback(CPropertyExtractor *cb_this, void *sp_ldos, const Index &index, int offset){
	int spinIndex = ((int*)(cb_this->hint))[0];
	Index to(index);
	Index from(index);
	complex<double> *greensFunction;

	for(int n = 0; n < 4; n++){
		to.indices.at(spinIndex) = n/2;		//up, up, down, down
		from.indices.at(spinIndex) = n%2;	//up, down, up, down
		greensFunction = cb_this->calculateGreensFunction(to, from);

		for(int e = 0; e < cb_this->energyResolution; e++)
			((complex<double>*)sp_ldos)[4*cb_this->energyResolution*offset + 4*e + n] = greensFunction[e];

		delete [] greensFunction;
	}
}

void CPropertyExtractor::calculate(void (*callback)(CPropertyExtractor *cb_this, void *memory, const Index &index, int offset),
					void *memory, Index pattern, const Index &ranges, int currentOffset, int offsetMultiplier){
	int currentSubindex = pattern.indices.size()-1;
	for(; currentSubindex >= 0; currentSubindex--){
		if(pattern.indices.at(currentSubindex) < 0)
			break;
	}

	if(currentSubindex == -1){
		callback(this, memory, pattern, currentOffset);
	}
	else{
		int nextOffsetMultiplier = offsetMultiplier;
		if(pattern.indices.at(currentSubindex) < IDX_SUM_ALL)
			nextOffsetMultiplier *= ranges.indices.at(currentSubindex);
		bool isSumIndex = false;
		if(pattern.indices.at(currentSubindex) == IDX_SUM_ALL)
			isSumIndex = true;
		for(int n = 0; n < ranges.indices.at(currentSubindex); n++){
			pattern.indices.at(currentSubindex) = n;
			calculate(callback,
					memory,
					pattern,
					ranges,
					currentOffset,
					nextOffsetMultiplier
			);
			if(!isSumIndex)
				currentOffset += offsetMultiplier;
		}
	}
}

};	//End of namespace TBTK
