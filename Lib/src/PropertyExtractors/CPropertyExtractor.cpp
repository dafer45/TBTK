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

/** @file CPropertyExtractor.cpp
 *
 *  @author Kristofer Björnson
 */

#include "CPropertyExtractor.h"
#include "Functions.h"
#include "TBTKMacros.h"
#include "Streams.h"

using namespace std;

namespace TBTK{

CPropertyExtractor::CPropertyExtractor(
	ChebyshevSolver *cSolver,
	int numCoefficients,
	int energyResolution,
	bool useGPUToCalculateCoefficients,
	bool useGPUToGenerateGreensFunctions,
	bool useLookupTable,
	double lowerBound,
	double upperBound
){
	TBTKAssert(
		cSolver != NULL,
		"CPropertyExtractor::CPropertyExtractor()",
		"Argument cSolver cannot be NULL.",
		""
	);
	TBTKAssert(
		numCoefficients > 0,
		"CPropertyExtractor::CPropertyExtractor()",
		"Argument numCoefficients has to be a positive number.",
		""
	);
	TBTKAssert(
		energyResolution > 0,
		"CPropertyExtractor::CPropertyExtractor()",
		"Argument energyResolution has to be a positive number.",
		""
	);
	TBTKAssert(
		lowerBound < upperBound,
		"CPropertyExtractor::CPropertyExtractor()",
		"Argument lowerBound has to be smaller than argument upperBound.",
		""
	);
	TBTKAssert(
		lowerBound >= -cSolver->getScaleFactor(),
		"CPropertyExtractor::CPropertyExtractor()",
		"Argument lowerBound has to be larger than -cSolver->getScaleFactor().",
		"Use ChebyshevSolver::setScaleFactor() to set a larger scale factor."
	);
	TBTKAssert(
		upperBound <= cSolver->getScaleFactor(),
		"CPropertyExtractor::CPropertyExtractor()",
		"Argument upperBound has to be smaller than cSolver->getScaleFactor().",
		"Use ChebyshevSolver::setScaleFactor() to set a larger scale factor."
	);

	this->cSolver = cSolver;
	this->numCoefficients = numCoefficients;
	this->energyResolution = energyResolution;
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->useGPUToCalculateCoefficients = useGPUToCalculateCoefficients;
	this->useGPUToGenerateGreensFunctions = useGPUToGenerateGreensFunctions;
	this->useLookupTable = useLookupTable;

	if(useLookupTable){
		cSolver->generateLookupTable(numCoefficients, energyResolution, lowerBound, upperBound);
		if(useGPUToGenerateGreensFunctions)
			cSolver->loadLookupTableGPU();
	}
	else if(useGPUToGenerateGreensFunctions){
		TBTKExit(
			"CPropertyExtractor::CPropertyExtractor()",
			"Argument 'useLookupTable' cannot be false if argument 'useGPUToGenerateGreensFunction' is true.",
			""
		);
	}
}

CPropertyExtractor::~CPropertyExtractor(){
	if(useGPUToGenerateGreensFunctions)
		cSolver->destroyLookupTableGPU();
}

complex<double>* CPropertyExtractor::calculateGreensFunction(
	Index to,
	Index from,
	ChebyshevSolver::GreensFunctionType type
){
	vector<Index> toIndices;
	toIndices.push_back(to);

	return calculateGreensFunctions(toIndices, from, type);
}

complex<double>* CPropertyExtractor::calculateGreensFunctions(
	vector<Index> &to,
	Index from,
	ChebyshevSolver::GreensFunctionType type
){
	complex<double> *coefficients = new complex<double>[numCoefficients*to.size()];

	if(useGPUToCalculateCoefficients){
		cSolver->calculateCoefficientsGPU(to, from, coefficients, numCoefficients);
	}
	else{
		cSolver->calculateCoefficients(to, from, coefficients, numCoefficients);
	}

	complex<double> *greensFunction = new complex<double>[energyResolution*to.size()];

	if(useGPUToGenerateGreensFunctions){
		for(unsigned int n = 0; n < to.size(); n++){
			cSolver->generateGreensFunctionGPU(greensFunction,
								&(coefficients[n*numCoefficients]),
								type);
		}
	}
	else{
		if(useLookupTable){
			#pragma omp parallel for
			for(unsigned int n = 0; n < to.size(); n++){
				cSolver->generateGreensFunction(greensFunction,
								&(coefficients[n*numCoefficients]),
								type);
			}
		}
		else{
			#pragma omp parallel for
			for(unsigned int n = 0; n < to.size(); n++){
				cSolver->generateGreensFunction(greensFunction,
								&(coefficients[n*numCoefficients]),
								numCoefficients,
								energyResolution,
								lowerBound,
								upperBound,
								type);
			}
		}
	}

	delete [] coefficients;

	return greensFunction;
}

complex<double> CPropertyExtractor::calculateExpectationValue(
	Index to,
	Index from
){
	const complex<double> i(0, 1);

	complex<double> expectationValue = 0.;

	complex<double> *greensFunction = calculateGreensFunction(to, from, ChebyshevSolver::GreensFunctionType::NonPrincipal);
	Model::Statistics statistics = cSolver->getModel()->getStatistics();

	const double dE = (upperBound - lowerBound)/energyResolution;
	for(int e = 0; e < energyResolution; e++){
		double weight;
		if(statistics == Model::Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(lowerBound + (e/(double)energyResolution)*(upperBound - lowerBound),
									cSolver->getModel()->getChemicalPotential(),
									cSolver->getModel()->getTemperature());
		}
		else{
			weight = Functions::boseEinsteinDistribution(lowerBound + (e/(double)energyResolution)*(upperBound - lowerBound),
									cSolver->getModel()->getChemicalPotential(),
									cSolver->getModel()->getTemperature());
		}

		expectationValue -= weight*conj(i*greensFunction[e])*dE/M_PI;
	}

	delete [] greensFunction;

	return expectationValue;
}

/*double* CPropertyExtractor::calculateDensity(Index pattern, Index ranges){
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
}*/

Property::Density* CPropertyExtractor::calculateDensity(
	Index pattern,
	Index ranges
){
	ensureCompliantRanges(pattern, ranges);

	int lDimensions = 0;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Density *density = new Property::Density(lDimensions, lRanges);

	calculate(calculateDensityCallback, (void*)density->data, pattern, ranges, 0, 1);

	return density;
}

Property::Magnetization* CPropertyExtractor::calculateMagnetization(
	Index pattern,
	Index ranges
){
	hint = new int[1];
	((int*)hint)[0] = -1;
	for(unsigned int n = 0; n < pattern.size(); n++){
		if(pattern.at(n) == IDX_SPIN){
			((int*)hint)[0] = n;
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(((int*)hint)[0] == -1){
		Streams::err << "Error in PropertyExtractorChebyshev::calculateMAG: No spin index indicated.\n";
		delete [] (int*)hint;
		return NULL;
	}

	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Magnetization *magnetization = new Property::Magnetization(lDimensions, lRanges);

	calculate(calculateMAGCallback, (void*)magnetization->data, pattern, ranges, 0, 1);

	delete [] (int*)hint;

	return magnetization;
}

/*double* CPropertyExtractor::calculateLDOS(Index pattern, Index ranges){
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
}*/

Property::LDOS* CPropertyExtractor::calculateLDOS(Index pattern, Index ranges){
	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::LDOS *ldos = new Property::LDOS(lDimensions, lRanges, lowerBound, upperBound, energyResolution);

	calculate(calculateLDOSCallback, (void*)ldos->data, pattern, ranges, 0, 1);

	return ldos;
}

Property::SpinPolarizedLDOS* CPropertyExtractor::calculateSpinPolarizedLDOS(
	Index pattern,
	Index ranges
){
	hint = new int[1];
	((int*)hint)[0] = -1;
	for(unsigned int n = 0; n < pattern.size(); n++){
		if(pattern.at(n) == IDX_SPIN){
			((int*)hint)[0] = n;
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(((int*)hint)[0] == -1){
		Streams::err << "Error in PropertyExtractorChebyshev::calculateSP_LDOS: No spin index indicated.\n";
		delete [] (int*)hint;
		return NULL;
	}

	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::SpinPolarizedLDOS *spinPolarizedLDOS = new Property::SpinPolarizedLDOS(lDimensions, lRanges, lowerBound, upperBound, energyResolution);

	calculate(calculateSP_LDOSCallback, (void*)spinPolarizedLDOS->data, pattern, ranges, 0, 1);

	delete [] (int*)hint;

	return spinPolarizedLDOS;
}

void CPropertyExtractor::calculateDensityCallback(
	PropertyExtractor *cb_this,
	void *density,
	const Index &index,
	int offset
){
	CPropertyExtractor *pe = (CPropertyExtractor*)cb_this;

	complex<double> *greensFunction = pe->calculateGreensFunction(index, index, ChebyshevSolver::GreensFunctionType::NonPrincipal);
	Model::Statistics statistics = pe->cSolver->getModel()->getStatistics();

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int e = 0; e < pe->energyResolution; e++){
		double weight;
		if(statistics == Model::Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
									pe->cSolver->getModel()->getChemicalPotential(),
									pe->cSolver->getModel()->getTemperature());
		}
		else{
			weight = Functions::boseEinsteinDistribution(pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
									pe->cSolver->getModel()->getChemicalPotential(),
									pe->cSolver->getModel()->getTemperature());
		}

		((double*)density)[offset] += weight*imag(greensFunction[e])/M_PI*dE;
	}

	delete [] greensFunction;
}

void CPropertyExtractor::calculateMAGCallback(
	PropertyExtractor *cb_this,
	void *mag,
	const Index &index,
	int offset
){
	CPropertyExtractor *pe = (CPropertyExtractor*)cb_this;

	int spinIndex = ((int*)(pe->hint))[0];
	Index to(index);
	Index from(index);
	complex<double> *greensFunction;
	Model::Statistics statistics = pe->cSolver->getModel()->getStatistics();

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;		//up, up, down, down
		from.at(spinIndex) = n%2;	//up, down, up, down
		greensFunction = pe->calculateGreensFunction(to, from, ChebyshevSolver::GreensFunctionType::NonPrincipal);

		for(int e = 0; e < pe->energyResolution; e++){
			double weight;
			if(statistics == Model::Statistics::FermiDirac){
				weight = Functions::fermiDiracDistribution(pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
										pe->cSolver->getModel()->getChemicalPotential(),
										pe->cSolver->getModel()->getTemperature());
			}
			else{
				weight = Functions::boseEinsteinDistribution(pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
										pe->cSolver->getModel()->getChemicalPotential(),
										pe->cSolver->getModel()->getTemperature());
			}

			((complex<double>*)mag)[4*offset + n] += weight*imag(greensFunction[e])/M_PI*dE;
		}

		delete [] greensFunction;
	}
}

void CPropertyExtractor::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	void *ldos,
	const Index &index,
	int offset
){
	CPropertyExtractor *pe = (CPropertyExtractor*)cb_this;

	complex<double> *greensFunction = pe->calculateGreensFunction(index, index, ChebyshevSolver::GreensFunctionType::NonPrincipal);

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < pe->energyResolution; n++)
		((double*)ldos)[pe->energyResolution*offset + n] += imag(greensFunction[n])/M_PI*dE;

	delete [] greensFunction;
}

void CPropertyExtractor::calculateSP_LDOSCallback(
	PropertyExtractor *cb_this,
	void *sp_ldos,
	const Index &index,
	int offset
){
	CPropertyExtractor *pe = (CPropertyExtractor*)cb_this;

	int spinIndex = ((int*)(pe->hint))[0];
	Index to(index);
	Index from(index);
	complex<double> *greensFunction;

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;		//up, up, down, down
		from.at(spinIndex) = n%2;	//up, down, up, down
		greensFunction = pe->calculateGreensFunction(to, from, ChebyshevSolver::GreensFunctionType::NonPrincipal);

		for(int e = 0; e < pe->energyResolution; e++)
			((complex<double>*)sp_ldos)[4*pe->energyResolution*offset + 4*e + n] += imag(greensFunction[e])/M_PI*dE;

		delete [] greensFunction;
	}
}

/*void CPropertyExtractor::calculate(
	void (*callback)(
		CPropertyExtractor *cb_this,
		void *memory,
		const Index &index,
		int offset
	),
	void *memory,
	Index pattern,
	const Index &ranges,
	int currentOffset,
	int offsetMultiplier
){
	int currentSubindex = pattern.size()-1;
	for(; currentSubindex >= 0; currentSubindex--){
		if(pattern.at(currentSubindex) < 0)
			break;
	}

	if(currentSubindex == -1){
		callback(this, memory, pattern, currentOffset);
	}
	else{
		int nextOffsetMultiplier = offsetMultiplier;
		if(pattern.at(currentSubindex) < IDX_SUM_ALL)
			nextOffsetMultiplier *= ranges.at(currentSubindex);
		bool isSumIndex = false;
		if(pattern.at(currentSubindex) == IDX_SUM_ALL)
			isSumIndex = true;
		for(int n = 0; n < ranges.at(currentSubindex); n++){
			pattern.at(currentSubindex) = n;
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

void CPropertyExtractor::ensureCompliantRanges(
	const Index &pattern,
	Index &ranges
){
	for(unsigned int n = 0; n < pattern.size(); n++){
		if(pattern.at(n) >= 0)
			ranges.at(n) = 1;
	}
}

void CPropertyExtractor::getLoopRanges(
	const Index &pattern,
	const Index &ranges,
	int *lDimensions,
	int **lRanges
){
	*lDimensions = 0;
	for(unsigned int n = 0; n < ranges.size(); n++){
		if(pattern.at(n) < IDX_SUM_ALL)
			(*lDimensions)++;
	}

	(*lRanges) = new int[*lDimensions];
	int counter = 0;
	for(unsigned int n = 0; n < ranges.size(); n++){
		if(pattern.at(n) < IDX_SUM_ALL)
			(*lRanges)[counter++] = ranges.at(n);
	}
}*/

};	//End of namespace TBTK
