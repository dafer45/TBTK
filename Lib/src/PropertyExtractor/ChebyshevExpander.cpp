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

#include "TBTK/PropertyExtractor/ChebyshevExpander.h"
#include "TBTK/Functions.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <set>

using namespace std;

static complex<double> i(0, 1);

namespace TBTK{
namespace PropertyExtractor{

ChebyshevExpander::ChebyshevExpander(
	Solver::ChebyshevExpander &cSolver,
	int numCoefficients,
	bool useGPUToCalculateCoefficients,
	bool useGPUToGenerateGreensFunctions,
	bool useLookupTable
){
	TBTKAssert(
		numCoefficients > 0,
		"PropertyExtractor::ChebyshevExpnader::ChebyshevExpander()",
		"Argument numCoefficients has to be a positive number.",
		""
	);
	TBTKAssert(
		energyResolution > 0,
		"PropertyExtractor::ChebyshevExpander::ChebyshevExpander()",
		"Argument energyResolution has to be a positive number.",
		""
	);
	TBTKAssert(
		lowerBound < upperBound,
		"PropertyExtractor::ChebyshevExpander::ChebyshevExpander()",
		"Argument lowerBound has to be smaller than argument upperBound.",
		""
	);
	TBTKAssert(
		lowerBound >= -cSolver.getScaleFactor(),
		"PropertyExtractor::ChebyshevExpander::ChebyshevExpander()",
		"Argument lowerBound has to be larger than -cSolver->getScaleFactor().",
		"Use Solver::ChebyshevExpander::setScaleFactor() to set a larger scale factor."
	);
	TBTKAssert(
		upperBound <= cSolver.getScaleFactor(),
		"PropertyExtractor::ChebyshevExapnder::ChebysheExpander()",
		"Argument upperBound has to be smaller than cSolver->getScaleFactor().",
		"Use Solver::ChebyshevExpnader::setScaleFactor() to set a larger scale factor."
	);

	this->cSolver = &cSolver;
	this->numCoefficients = numCoefficients;
	this->useGPUToCalculateCoefficients = useGPUToCalculateCoefficients;
	this->useGPUToGenerateGreensFunctions = useGPUToGenerateGreensFunctions;
	this->useLookupTable = useLookupTable;

	setEnergyWindow(
		-cSolver.getScaleFactor(),
		cSolver.getScaleFactor(),
		ENERGY_RESOLUTION
	);

}

ChebyshevExpander::~ChebyshevExpander(){
	if(useGPUToGenerateGreensFunctions)
		cSolver->destroyLookupTableGPU();
}

void ChebyshevExpander::setEnergyWindow(
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

//Property::GreensFunction* CPropertyExtractor::calculateGreensFunction(
Property::GreensFunction ChebyshevExpander::calculateGreensFunction(
	Index to,
	Index from,
	Property::GreensFunction::Type type
){
	vector<Index> toIndices;
	toIndices.push_back(to);

	return calculateGreensFunctions(toIndices, from, type);
}

Property::GreensFunction ChebyshevExpander::calculateGreensFunction(
	initializer_list<initializer_list<Index>> patterns,
	Property::GreensFunction::Type type
){
	IndexTree memoryLayout;
	IndexTree fromIndices;
	set<unsigned int> toIndexSizes;
	for(unsigned int n = 0; n < patterns.size(); n++){
		const initializer_list<Index>& pattern = *(patterns.begin() + n);

		TBTKAssert(
			pattern.size() == 2,
			"ChebyshevExpander::calculateGreensFunction()",
			"Invalid pattern. Each pattern entry needs to contain"
			" exactly two Indices, but '"
			<< pattern.size() << "' found in entry '" << n << "'.",
			""
		);

		Index toPattern = *(pattern.begin() + 0);
		Index fromPattern = *(pattern.begin() + 1);

		toIndexSizes.insert(toPattern.getSize());

		IndexTree toTree = generateIndexTree(
			{toPattern},
			*cSolver->getModel().getHoppingAmplitudeSet(),
			true,
			true
		);
		IndexTree fromTree = generateIndexTree(
			{fromPattern},
			*cSolver->getModel().getHoppingAmplitudeSet(),
			true,
			true
		);

		IndexTree::Iterator toIterator = toTree.begin();
		while(!toIterator.getHasReachedEnd()){
			Index toIndex = toIterator.getIndex();
			IndexTree::Iterator fromIterator = fromTree.begin();
			while(!fromIterator.getHasReachedEnd()){
				Index fromIndex = fromIterator.getIndex();
				memoryLayout.add({toIndex, fromIndex});
				fromIndices.add(fromIndex);

				fromIterator.searchNext();
			}

			toIterator.searchNext();
		}
	}
	memoryLayout.generateLinearMap();
	fromIndices.generateLinearMap();

	Property::GreensFunction greensFunction(
		memoryLayout,
		type,
		lowerBound,
		upperBound,
		energyResolution
	);

	IndexTree::Iterator iterator = fromIndices.begin();
	while(!iterator.getHasReachedEnd()){
		Index fromIndex = iterator.getIndex();
		vector<Index> toIndices;
		for(
			set<unsigned int>::iterator iterator = toIndexSizes.begin();
			iterator != toIndexSizes.end();
			++iterator
		){
			Index toPattern;
			for(unsigned int n = 0; n < *iterator; n++){
				toPattern.push_back(IDX_ALL);
			}

			vector<Index> matchingIndices = memoryLayout.getIndexList(
				{toPattern, fromIndex}
			);
			for(unsigned int n = 0; n < matchingIndices.size(); n++){
				Index toIndex;
				for(unsigned int c = 0; c < *iterator; c++)
					toIndex.push_back(matchingIndices[n][c]);
				toIndices.push_back(toIndex);
			}
		}

		Property::GreensFunction gf = calculateGreensFunctions(
			toIndices,
			fromIndex,
			type
		);

		for(unsigned int n = 0; n < toIndices.size(); n++){
			complex<double> *data = greensFunction.getDataRW();
			const complex<double> *dataGF = gf.getData();

			Index compoundIndex = Index({toIndices[n], fromIndex});

			unsigned int offset = greensFunction.getOffset(
				compoundIndex
			);
			unsigned int offsetGF = gf.getOffset(
				compoundIndex
			);

			for(int c = 0; c < energyResolution; c++){
				data[offset + c] = dataGF[offsetGF + c];
			}
		}

		iterator.searchNext();
	}

	return greensFunction;
}

Property::GreensFunction ChebyshevExpander::calculateGreensFunctions(
	vector<Index> &to,
	Index from,
	Property::GreensFunction::Type type
){
	ensureLookupTableIsReady();

	complex<double> *coefficients = new complex<double>[numCoefficients*to.size()];

/*	if(useGPUToCalculateCoefficients){
		cSolver->calculateCoefficientsGPU(to, from, coefficients, numCoefficients);
	}
	else{
		cSolver->calculateCoefficientsCPU(to, from, coefficients, numCoefficients);
	}*/
	cSolver->calculateCoefficients(to, from, coefficients/*, numCoefficients*/);

	Solver::ChebyshevExpander::Type chebyshevType;
	switch(type){
	case Property::GreensFunction::Type::Advanced:
		chebyshevType = Solver::ChebyshevExpander::Type::Advanced;
		break;
	case Property::GreensFunction::Type::Retarded:
		chebyshevType = Solver::ChebyshevExpander::Type::Retarded;
		break;
	case Property::GreensFunction::Type::Principal:
		chebyshevType = Solver::ChebyshevExpander::Type::Principal;
		break;
	case Property::GreensFunction::Type::NonPrincipal:
		chebyshevType = Solver::ChebyshevExpander::Type::NonPrincipal;
		break;
	default:
		TBTKExit(
			"PropertyExtractor::ChebyshevExpander::calculateGreensFunctions()",
			"Unknown GreensFunction type.",
			"This should never happen, contact the developer."
		);
	}

	IndexTree memoryLayout;
	for(unsigned int n = 0; n < to.size(); n++)
		memoryLayout.add({to[n], from});
	memoryLayout.generateLinearMap();
	Property::GreensFunction greensFunction(
		memoryLayout,
		type,
		lowerBound,
		upperBound,
		energyResolution
	);
	complex<double> *data = greensFunction.getDataRW();

	if(useGPUToGenerateGreensFunctions){
		for(unsigned int n = 0; n < to.size(); n++){
			complex<double> *greensFunctionData = cSolver->generateGreensFunctionGPU(
				&(coefficients[n*numCoefficients]),
				chebyshevType
			);
			unsigned int offset = greensFunction.getOffset({to[n], from});
			for(int c = 0; c < energyResolution; c++)
				data[offset + c] = greensFunctionData[c];
			delete [] greensFunctionData;
		}
	}
	else{
		if(useLookupTable){
			#pragma omp parallel for
			for(unsigned int n = 0; n < to.size(); n++){
				complex<double> *greensFunctionData = cSolver->generateGreensFunctionCPU(
					&(coefficients[n*numCoefficients]),
					chebyshevType
				);
				unsigned int offset = greensFunction.getOffset({to[n], from});
				for(int c = 0; c < energyResolution; c++)
					data[offset + c] = greensFunctionData[c];
				delete [] greensFunctionData;
			}
		}
		else{
			#pragma omp parallel for
			for(unsigned int n = 0; n < to.size(); n++){
				complex<double> *greensFunctionData = cSolver->generateGreensFunctionCPU(
					&(coefficients[n*numCoefficients]),
					numCoefficients,
					energyResolution,
					lowerBound,
					upperBound,
					chebyshevType
				);
				unsigned int offset = greensFunction.getOffset({to[n], from});
				for(int c = 0; c < energyResolution; c++)
					data[offset + c] = greensFunctionData[c];
				delete [] greensFunctionData;
			}
		}
	}

	delete [] coefficients;

	return greensFunction;
}

complex<double> ChebyshevExpander::calculateExpectationValue(
	Index to,
	Index from
){
	const complex<double> i(0, 1);

	complex<double> expectationValue = 0.;

	Property::GreensFunction greensFunction = calculateGreensFunction(
		to,
		from,
		Property::GreensFunction::Type::NonPrincipal
	);
	const complex<double> *greensFunctionData = greensFunction.getData();

	Statistics statistics = cSolver->getModel().getStatistics();

	const double dE = (upperBound - lowerBound)/energyResolution;
	for(int e = 0; e < energyResolution; e++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				lowerBound + (e/(double)energyResolution)*(upperBound - lowerBound),
				cSolver->getModel().getChemicalPotential(),
				cSolver->getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				lowerBound + (e/(double)energyResolution)*(upperBound - lowerBound),
				cSolver->getModel().getChemicalPotential(),
				cSolver->getModel().getTemperature()
			);
		}

		expectationValue -= weight*conj(i*greensFunctionData[e])*dE/M_PI;
	}

	return expectationValue;
}

Property::Density ChebyshevExpander::calculateDensity(
	Index pattern,
	Index ranges
){
	ensureCompliantRanges(pattern, ranges);

	int lDimensions = 0;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Density density(lDimensions, lRanges);

	calculate(
		calculateDensityCallback,
		(void*)density.getDataRW(),
		pattern,
		ranges,
		0,
		1
	);

	return density;
}

Property::Density ChebyshevExpander::calculateDensity(
	std::initializer_list<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		*cSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*cSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::Density density(memoryLayout);

	calculate(
		calculateDensityCallback,
		allIndices,
		memoryLayout,
		density
	);

	return density;
}

Property::Magnetization ChebyshevExpander::calculateMagnetization(
	Index pattern,
	Index ranges
){
	hint = new int[1];
	((int*)hint)[0] = -1;
	for(unsigned int n = 0; n < pattern.getSize(); n++){
		if(pattern.at(n) == IDX_SPIN){
			((int*)hint)[0] = n;
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(((int*)hint)[0] == -1){
		delete [] (int*)hint;
		TBTKExit(
			"PropertyExtractor::ChebyshevExpander::calculateMagnetization()",
			"No spin index indicated.",
			"Use IDX_SPIN to indicate the position of the spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Magnetization magnetization(lDimensions, lRanges);

	calculate(
		calculateMAGCallback,
		(void*)magnetization.getDataRW(),
		pattern,
		ranges,
		0,
		1
	);

	delete [] (int*)hint;

	return magnetization;
}

Property::Magnetization ChebyshevExpander::calculateMagnetization(
	std::initializer_list<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		*cSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*cSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::Magnetization magnetization(memoryLayout);

	hint = new int[1];
	calculate(
		calculateMAGCallback,
		allIndices,
		memoryLayout,
		magnetization,
		(int*)hint
	);

	delete [] (int*)hint;

	return magnetization;
}

Property::LDOS ChebyshevExpander::calculateLDOS(Index pattern, Index ranges){
	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::LDOS ldos(
		lDimensions,
		lRanges,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateLDOSCallback,
		(void*)ldos.getDataRW(),
		pattern,
		ranges,
		0,
		/*1*/energyResolution
	);

	return ldos;
}

Property::LDOS ChebyshevExpander::calculateLDOS(
	std::initializer_list<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		*cSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*cSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::LDOS ldos(
		memoryLayout,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateLDOSCallback,
		allIndices,
		memoryLayout,
		ldos
	);

	return ldos;
}

Property::SpinPolarizedLDOS ChebyshevExpander::calculateSpinPolarizedLDOS(
	Index pattern,
	Index ranges
){
	hint = new int[1];
	((int*)hint)[0] = -1;
	for(unsigned int n = 0; n < pattern.getSize(); n++){
		if(pattern.at(n) == IDX_SPIN){
			((int*)hint)[0] = n;
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(((int*)hint)[0] == -1){
		delete [] (int*)hint;
		TBTKExit(
			"PropertyExtractor::ChebsyhevExpander::calculateSpinPolarizedLDOS()",
			"No spin index indicated.",
			"Use IDX_SPIN to indicate the position of the spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		lDimensions,
		lRanges,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateSP_LDOSCallback,
		(void*)spinPolarizedLDOS.getDataRW(),
		pattern,
		ranges,
		0,
		energyResolution
	);

	delete [] (int*)hint;

	return spinPolarizedLDOS;
}

Property::SpinPolarizedLDOS ChebyshevExpander::calculateSpinPolarizedLDOS(
	std::initializer_list<Index> patterns
){
	hint = new int[1];

	IndexTree allIndices = generateIndexTree(
		patterns,
		*cSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		*cSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		memoryLayout,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateSP_LDOSCallback,
		allIndices,
		memoryLayout,
		spinPolarizedLDOS,
		(int*)hint
	);

	delete [] (int*)hint;

	return spinPolarizedLDOS;
}

void ChebyshevExpander::calculateDensityCallback(
	PropertyExtractor *cb_this,
	void *density,
	const Index &index,
	int offset
){
	ChebyshevExpander *pe = (ChebyshevExpander*)cb_this;

	Property::GreensFunction greensFunction = pe->calculateGreensFunction(
		index,
		index,
		Property::GreensFunction::Type::NonPrincipal
	);
	const complex<double> *greensFunctionData = greensFunction.getData();

	Statistics statistics = pe->cSolver->getModel().getStatistics();

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int e = 0; e < pe->energyResolution; e++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
				pe->cSolver->getModel().getChemicalPotential(),
				pe->cSolver->getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
				pe->cSolver->getModel().getChemicalPotential(),
				pe->cSolver->getModel().getTemperature()
			);
		}

		((double*)density)[offset] += weight*imag(greensFunctionData[e])/M_PI*dE;
	}
}

void ChebyshevExpander::calculateMAGCallback(
	PropertyExtractor *cb_this,
	void *mag,
	const Index &index,
	int offset
){
	ChebyshevExpander *pe = (ChebyshevExpander*)cb_this;

	int spinIndex = ((int*)(pe->hint))[0];
	Index to(index);
	Index from(index);
	Statistics statistics = pe->cSolver->getModel().getStatistics();

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;		//up, up, down, down
		from.at(spinIndex) = n%2;	//up, down, up, down
		Property::GreensFunction greensFunction = pe->calculateGreensFunction(
			to,
			from,
			Property::GreensFunction::Type::NonPrincipal
		);
		const complex<double> *greensFunctionData = greensFunction.getData();

		for(int e = 0; e < pe->energyResolution; e++){
			double weight;
			if(statistics == Statistics::FermiDirac){
				weight = Functions::fermiDiracDistribution(
					pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
					pe->cSolver->getModel().getChemicalPotential(),
					pe->cSolver->getModel().getTemperature()
				);
			}
			else{
				weight = Functions::boseEinsteinDistribution(
					pe->lowerBound + (e/(double)pe->energyResolution)*(pe->upperBound - pe->lowerBound),
					pe->cSolver->getModel().getChemicalPotential(),
					pe->cSolver->getModel().getTemperature()
				);
			}

			((SpinMatrix*)mag)[offset].at(n/2, n%2) += weight*(-i)*greensFunctionData[e]/M_PI*dE;
		}
	}
}

void ChebyshevExpander::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	void *ldos,
	const Index &index,
	int offset
){
	ChebyshevExpander *pe = (ChebyshevExpander*)cb_this;

	Property::GreensFunction greensFunction = pe->calculateGreensFunction(
		index,
		index,
		Property::GreensFunction::Type::NonPrincipal
	);
	const complex<double> *greensFunctionData = greensFunction.getData();

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < pe->energyResolution; n++)
		((double*)ldos)[offset + n] += imag(greensFunctionData[n])/M_PI*dE;
}

void ChebyshevExpander::calculateSP_LDOSCallback(
	PropertyExtractor *cb_this,
	void *sp_ldos,
	const Index &index,
	int offset
){
	ChebyshevExpander *pe = (ChebyshevExpander*)cb_this;

	int spinIndex = ((int*)(pe->hint))[0];
	Index to(index);
	Index from(index);

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;		//up, up, down, down
		from.at(spinIndex) = n%2;	//up, down, up, down
		Property::GreensFunction greensFunction = pe->calculateGreensFunction(
			to,
			from,
			Property::GreensFunction::Type::NonPrincipal
		);
		const complex<double> *greensFunctionData = greensFunction.getData();

		for(int e = 0; e < pe->energyResolution; e++)
			((SpinMatrix*)sp_ldos)[offset + e].at(n/2, n%2) += -i*greensFunctionData[e]/M_PI*dE;
	}
}

void ChebyshevExpander::ensureLookupTableIsReady(){
	if(useLookupTable){
		if(!cSolver->getLookupTableIsGenerated())
			cSolver->generateLookupTable(numCoefficients, energyResolution, lowerBound, upperBound);
		if(useGPUToGenerateGreensFunctions && !cSolver->getLookupTableIsLoadedGPU())
			cSolver->loadLookupTableGPU();
	}
	else if(useGPUToGenerateGreensFunctions){
		TBTKExit(
			"PropertyExtractor::ChebyshevExpander::ensureLookupTableIsReady()",
			"Argument 'useLookupTable' cannot be false if argument"
			<< " 'useGPUToGenerateGreensFunction' is true.",
			""
		);
	}
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
