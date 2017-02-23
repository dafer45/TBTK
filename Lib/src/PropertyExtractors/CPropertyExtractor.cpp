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
#include "Streams.h"
#include "TBTKMacros.h"

using namespace std;

static complex<double> i(0, 1);

namespace TBTK{

CPropertyExtractor::CPropertyExtractor(
	ChebyshevSolver *cSolver,
	int numCoefficients,
	bool useGPUToCalculateCoefficients,
	bool useGPUToGenerateGreensFunctions,
	bool useLookupTable
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
	this->useGPUToCalculateCoefficients = useGPUToCalculateCoefficients;
	this->useGPUToGenerateGreensFunctions = useGPUToGenerateGreensFunctions;
	this->useLookupTable = useLookupTable;

	setEnergyWindow(
		-cSolver->getScaleFactor(),
		cSolver->getScaleFactor(),
		ENERGY_RESOLUTION
	);

}

CPropertyExtractor::~CPropertyExtractor(){
	if(useGPUToGenerateGreensFunctions)
		cSolver->destroyLookupTableGPU();
}

void CPropertyExtractor::setEnergyWindow(
	double lowerBound,
	double upperBound,
	int energyResolution
){
	PropertyExtractor::setEnergyWindow(
		lowerBound,
		upperBound,
		energyResolution
	);

	if(cSolver->getLookupTableIsLoadedGPU())
		cSolver->destroyLookupTableGPU();
	if(cSolver->getLookupTableIsGenerated())
		cSolver->destroyLookupTable();
}

Property::GreensFunction* CPropertyExtractor::calculateGreensFunction(
	Index to,
	Index from,
	Property::GreensFunction::Type type
){
	vector<Index> toIndices;
	toIndices.push_back(to);

	Property::GreensFunction **greensFunctions = calculateGreensFunctions(toIndices, from, type);
	Property::GreensFunction *greensFunction = greensFunctions[0];
	delete [] greensFunctions;

	return greensFunction;
}

Property::GreensFunction** CPropertyExtractor::calculateGreensFunctions(
	vector<Index> &to,
	Index from,
	Property::GreensFunction::Type type
){
	ensureLookupTableIsReady();

	complex<double> *coefficients = new complex<double>[numCoefficients*to.size()];

	if(useGPUToCalculateCoefficients){
		cSolver->calculateCoefficientsGPU(to, from, coefficients, numCoefficients);
	}
	else{
		cSolver->calculateCoefficients(to, from, coefficients, numCoefficients);
	}

	Property::GreensFunction **greensFunctions = new Property::GreensFunction*[to.size()];

	if(useGPUToGenerateGreensFunctions){
		for(unsigned int n = 0; n < to.size(); n++){
			greensFunctions[n] = cSolver->generateGreensFunctionGPU(
				&(coefficients[n*numCoefficients]),
				type
			);
		}
	}
	else{
		if(useLookupTable){
			#pragma omp parallel for
			for(unsigned int n = 0; n < to.size(); n++){
				greensFunctions[n] = cSolver->generateGreensFunction(
					&(coefficients[n*numCoefficients]),
					type
				);
			}
		}
		else{
			#pragma omp parallel for
			for(unsigned int n = 0; n < to.size(); n++){
				greensFunctions[n] = cSolver->generateGreensFunction(
					&(coefficients[n*numCoefficients]),
					numCoefficients,
					energyResolution,
					lowerBound,
					upperBound,
					type
				);
			}
		}
	}

	delete [] coefficients;

	return greensFunctions;
}

complex<double> CPropertyExtractor::calculateExpectationValue(
	Index to,
	Index from
){
	const complex<double> i(0, 1);

	complex<double> expectationValue = 0.;

	Property::GreensFunction *greensFunction = calculateGreensFunction(
		to,
		from,
		Property::GreensFunction::Type::NonPrincipal
	);
	const complex<double> *greensFunctionData = greensFunction->getArrayData();

	Statistics statistics = cSolver->getModel()->getStatistics();

	const double dE = (upperBound - lowerBound)/energyResolution;
	for(int e = 0; e < energyResolution; e++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				lowerBound + (e/(double)energyResolution)*(upperBound - lowerBound),
				cSolver->getModel()->getChemicalPotential(),
				cSolver->getModel()->getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				lowerBound + (e/(double)energyResolution)*(upperBound - lowerBound),
				cSolver->getModel()->getChemicalPotential(),
				cSolver->getModel()->getTemperature()
			);
		}

		expectationValue -= weight*conj(i*greensFunctionData[e])*dE/M_PI;
	}

	delete greensFunction;

	return expectationValue;
}

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

	Property::GreensFunction *greensFunction = pe->calculateGreensFunction(
		index,
		index,
		Property::GreensFunction::Type::NonPrincipal
	);
	const complex<double> *greensFunctionData = greensFunction->getArrayData();

	Statistics statistics = pe->cSolver->getModel()->getStatistics();

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int e = 0; e < pe->energyResolution; e++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
				pe->cSolver->getModel()->getChemicalPotential(),
				pe->cSolver->getModel()->getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
				pe->cSolver->getModel()->getChemicalPotential(),
				pe->cSolver->getModel()->getTemperature()
			);
		}

		((double*)density)[offset] += weight*imag(greensFunctionData[e])/M_PI*dE;
	}

	delete greensFunction;
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
	Statistics statistics = pe->cSolver->getModel()->getStatistics();

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;		//up, up, down, down
		from.at(spinIndex) = n%2;	//up, down, up, down
		Property::GreensFunction *greensFunction = pe->calculateGreensFunction(
			to,
			from,
			Property::GreensFunction::Type::NonPrincipal
		);
		const complex<double> *greensFunctionData = greensFunction->getArrayData();

		for(int e = 0; e < pe->energyResolution; e++){
			double weight;
			if(statistics == Statistics::FermiDirac){
				weight = Functions::fermiDiracDistribution(
					pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
					pe->cSolver->getModel()->getChemicalPotential(),
					pe->cSolver->getModel()->getTemperature()
				);
			}
			else{
				weight = Functions::boseEinsteinDistribution(
					pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
					pe->cSolver->getModel()->getChemicalPotential(),
					pe->cSolver->getModel()->getTemperature()
				);
			}

			((complex<double>*)mag)[4*offset + n] += weight*(-i)*greensFunctionData[e]/M_PI*dE;
		}

		delete greensFunction;
	}
}

void CPropertyExtractor::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	void *ldos,
	const Index &index,
	int offset
){
	CPropertyExtractor *pe = (CPropertyExtractor*)cb_this;

	Property::GreensFunction *greensFunction = pe->calculateGreensFunction(
		index,
		index,
		Property::GreensFunction::Type::NonPrincipal
	);
	const complex<double> *greensFunctionData = greensFunction->getArrayData();

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < pe->energyResolution; n++)
		((double*)ldos)[pe->energyResolution*offset + n] += imag(greensFunctionData[n])/M_PI*dE;

	delete greensFunction;
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

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;		//up, up, down, down
		from.at(spinIndex) = n%2;	//up, down, up, down
		Property::GreensFunction *greensFunction = pe->calculateGreensFunction(
			to,
			from,
			Property::GreensFunction::Type::NonPrincipal
		);
		const complex<double> *greensFunctionData = greensFunction->getArrayData();

		for(int e = 0; e < pe->energyResolution; e++)
			((complex<double>*)sp_ldos)[4*pe->energyResolution*offset + 4*e + n] += -i*greensFunctionData[e]/M_PI*dE;

		delete greensFunction;
	}
}

void CPropertyExtractor::ensureLookupTableIsReady(){
	if(useLookupTable){
		if(!cSolver->getLookupTableIsGenerated())
			cSolver->generateLookupTable(numCoefficients, energyResolution, lowerBound, upperBound);
		if(useGPUToGenerateGreensFunctions && !cSolver->getLookupTableIsLoadedGPU())
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

};	//End of namespace TBTK
