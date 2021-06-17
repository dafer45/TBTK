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

ChebyshevExpander::ChebyshevExpander(Solver::ChebyshevExpander &cSolver){
	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

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

	setEnergyWindow(
		-cSolver.getScaleFactor(),
		cSolver.getScaleFactor(),
		energyResolution
	);

}

ChebyshevExpander::~ChebyshevExpander(){
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

	cSolver->setLowerBound(lowerBound);
	cSolver->setUpperBound(upperBound);
	cSolver->setEnergyResolution(energyResolution);
}

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
	vector<vector<Index>> patterns,
	Property::GreensFunction::Type type
){
	IndexTree memoryLayout;
	IndexTree fromIndices;
	set<unsigned int> toIndexSizes;
	for(unsigned int n = 0; n < patterns.size(); n++){
		const vector<Index>& pattern = *(patterns.begin() + n);

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
			cSolver->getModel().getHoppingAmplitudeSet(),
			true,
			true
		);
		IndexTree fromTree = generateIndexTree(
			{fromPattern},
			cSolver->getModel().getHoppingAmplitudeSet(),
			true,
			true
		);

		for(
			IndexTree::ConstIterator toIterator = toTree.cbegin();
			toIterator != toTree.cend();
			++toIterator
		){
			Index toIndex = *toIterator;
			for(
				IndexTree::ConstIterator fromIterator
					= fromTree.cbegin();
				fromIterator != fromTree.cend();
				++fromIterator
			){
				Index fromIndex = *fromIterator;
				memoryLayout.add({toIndex, fromIndex});
				fromIndices.add(fromIndex);
			}
		}
	}
	memoryLayout.generateLinearMap();
	fromIndices.generateLinearMap();

	Property::GreensFunction greensFunction(
		memoryLayout,
		type,
		getLowerBound(),
		getUpperBound(),
		getEnergyResolution()
	);

	for(
		IndexTree::ConstIterator iterator = fromIndices.cbegin();
		iterator != fromIndices.cend();
		++iterator
	){
		Index fromIndex = *iterator;
		vector<Index> toIndices;
		for(
			set<unsigned int>::iterator iterator = toIndexSizes.begin();
			iterator != toIndexSizes.end();
			++iterator
		){
			Index toPattern;
			for(unsigned int n = 0; n < *iterator; n++){
				toPattern.pushBack(IDX_ALL);
			}

			vector<Index> matchingIndices = memoryLayout.getIndexList(
				{toPattern, fromIndex}
			);
			for(unsigned int n = 0; n < matchingIndices.size(); n++){
				Index toIndex;
				for(unsigned int c = 0; c < *iterator; c++)
					toIndex.pushBack(matchingIndices[n][c]);
				toIndices.push_back(toIndex);
			}
		}

		Property::GreensFunction gf = calculateGreensFunctions(
			toIndices,
			fromIndex,
			type
		);

		for(unsigned int n = 0; n < toIndices.size(); n++){
			std::vector<complex<double>> &data
				= greensFunction.getDataRW();
			const std::vector<complex<double>> &dataGF
				= gf.getData();

			Index compoundIndex = Index({toIndices[n], fromIndex});

			unsigned int offset = greensFunction.getOffset(
				compoundIndex
			);
			unsigned int offsetGF = gf.getOffset(
				compoundIndex
			);

			for(int c = 0; c < getEnergyResolution(); c++){
				data[offset + c] = dataGF[offsetGF + c];
			}
		}
	}

	return greensFunction;
}

Property::GreensFunction ChebyshevExpander::calculateGreensFunctions(
	vector<Index> &to,
	Index from,
	Property::GreensFunction::Type type
){
	vector<vector<complex<double>>> coefficients = cSolver->calculateCoefficients(
		to,
		from
	);

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
		getLowerBound(),
		getUpperBound(),
		getEnergyResolution()
	);
	std::vector<complex<double>> &data = greensFunction.getDataRW();

	#pragma omp parallel for
	for(unsigned int n = 0; n < to.size(); n++){
		vector<complex<double>> greensFunctionData = cSolver->generateGreensFunction(
			coefficients[n],
			chebyshevType
		);
		unsigned int offset = greensFunction.getOffset({to[n], from});
		for(int c = 0; c < getEnergyResolution(); c++)
			data[offset + c] = greensFunctionData[c];
	}

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
	const std::vector<complex<double>> &greensFunctionData
		= greensFunction.getData();

	Statistics statistics = cSolver->getModel().getStatistics();

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

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

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::Density density(loopRanges);

	Information information;
	calculate(
		calculateDensityCallback,
		density,
		pattern,
		ranges,
		0,
		1,
		information
	);

	return density;
}

Property::Density ChebyshevExpander::calculateDensity(
	vector<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		cSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		cSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::Density density(memoryLayout);

	Information information;
	calculate(
		calculateDensityCallback,
		allIndices,
		memoryLayout,
		density,
		information
	);

	return density;
}

Property::Magnetization ChebyshevExpander::calculateMagnetization(
	Index pattern,
	Index ranges
){
	Information information;
	for(unsigned int n = 0; n < pattern.getSize(); n++){
		if(pattern.at(n).isSpinIndex()){
			information.setSpinIndex(n);
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(information.getSpinIndex() == -1){
		TBTKExit(
			"PropertyExtractor::ChebyshevExpander::calculateMagnetization()",
			"No spin index indicated.",
			"Use IDX_SPIN to indicate the position of the spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::Magnetization magnetization(loopRanges);

	calculate(
		calculateMAGCallback,
		magnetization,
		pattern,
		ranges,
		0,
		1,
		information
	);

	return magnetization;
}

Property::Magnetization ChebyshevExpander::calculateMagnetization(
	vector<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		cSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		cSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::Magnetization magnetization(memoryLayout);

	Information information;
	calculate(
		calculateMAGCallback,
		allIndices,
		memoryLayout,
		magnetization,
		information
	);

	return magnetization;
}

Property::LDOS ChebyshevExpander::calculateLDOS(Index pattern, Index ranges){
	ensureCompliantRanges(pattern, ranges);

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::LDOS ldos(
		loopRanges,
		lowerBound,
		upperBound,
		energyResolution
	);

	Information information;
	calculate(
		calculateLDOSCallback,
		ldos,
		pattern,
		ranges,
		0,
		energyResolution,
		information
	);

	return ldos;
}

Property::LDOS ChebyshevExpander::calculateLDOS(
	vector<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		cSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		cSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::LDOS ldos(
		memoryLayout,
		getLowerBound(),
		getUpperBound(),
		getEnergyResolution()
	);

	Information information;
	calculate(
		calculateLDOSCallback,
		allIndices,
		memoryLayout,
		ldos,
		information
	);

	return ldos;
}

Property::SpinPolarizedLDOS ChebyshevExpander::calculateSpinPolarizedLDOS(
	Index pattern,
	Index ranges
){
	Information information;
	for(unsigned int n = 0; n < pattern.getSize(); n++){
		if(pattern.at(n).isSpinIndex()){
			information.setSpinIndex(n);
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(information.getSpinIndex() == -1){
		TBTKExit(
			"PropertyExtractor::ChebsyhevExpander::calculateSpinPolarizedLDOS()",
			"No spin index indicated.",
			"Use IDX_SPIN to indicate the position of the spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		loopRanges,
		getLowerBound(),
		getUpperBound(),
		getEnergyResolution()
	);

	calculate(
		calculateSP_LDOSCallback,
		spinPolarizedLDOS,
		pattern,
		ranges,
		0,
		getEnergyResolution(),
		information
	);

	return spinPolarizedLDOS;
}

Property::SpinPolarizedLDOS ChebyshevExpander::calculateSpinPolarizedLDOS(
	vector<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		cSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		cSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		memoryLayout,
		getLowerBound(),
		getUpperBound(),
		getEnergyResolution()
	);

	Information information;
	calculate(
		calculateSP_LDOSCallback,
		allIndices,
		memoryLayout,
		spinPolarizedLDOS,
		information
	);

	return spinPolarizedLDOS;
}

void ChebyshevExpander::calculateDensityCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	ChebyshevExpander *pe = (ChebyshevExpander*)cb_this;
	Property::Density &density = (Property::Density&)property;
	vector<double> &data = density.getDataRW();

	Property::GreensFunction greensFunction = pe->calculateGreensFunction(
		index,
		index,
		Property::GreensFunction::Type::NonPrincipal
	);
	const std::vector<complex<double>> &greensFunctionData
		= greensFunction.getData();

	Statistics statistics = pe->cSolver->getModel().getStatistics();

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();

	const double dE = (upperBound - lowerBound)/energyResolution;
	for(int e = 0; e < energyResolution; e++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				lowerBound + (e/(double)energyResolution)*(upperBound - lowerBound),
				pe->cSolver->getModel().getChemicalPotential(),
				pe->cSolver->getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				lowerBound + (e/(double)energyResolution)*(upperBound - lowerBound),
				pe->cSolver->getModel().getChemicalPotential(),
				pe->cSolver->getModel().getTemperature()
			);
		}

		data[offset] += weight*imag(greensFunctionData[e])/M_PI*dE;
	}
}

void ChebyshevExpander::calculateMAGCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	ChebyshevExpander *pe = (ChebyshevExpander*)cb_this;
	Property::Magnetization &magnetization
		= (Property::Magnetization&)property;
	vector<SpinMatrix> &data = magnetization.getDataRW();

	int spinIndex = information.getSpinIndex();
	Index to(index);
	Index from(index);
	Statistics statistics = pe->cSolver->getModel().getStatistics();

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();

	const double dE = (upperBound - lowerBound)/energyResolution;
	for(int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;		//up, up, down, down
		from.at(spinIndex) = n%2;	//up, down, up, down
		Property::GreensFunction greensFunction
			= pe->calculateGreensFunction(
				to,
				from,
				Property::GreensFunction::Type::NonPrincipal
			);
		const std::vector<complex<double>> &greensFunctionData
			= greensFunction.getData();

		for(int e = 0; e < energyResolution; e++){
			double weight;
			if(statistics == Statistics::FermiDirac){
				weight = Functions::fermiDiracDistribution(
					lowerBound + (e/(double)energyResolution)*(upperBound - lowerBound),
					pe->cSolver->getModel().getChemicalPotential(),
					pe->cSolver->getModel().getTemperature()
				);
			}
			else{
				weight = Functions::boseEinsteinDistribution(
					lowerBound + (e/(double)energyResolution)*(upperBound - lowerBound),
					pe->cSolver->getModel().getChemicalPotential(),
					pe->cSolver->getModel().getTemperature()
				);
			}

			data[offset].at(n/2, n%2)
				+= weight*(-i)*greensFunctionData[e]/M_PI*dE;
		}
	}
}

void ChebyshevExpander::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	ChebyshevExpander *pe = (ChebyshevExpander*)cb_this;
	Property::LDOS &ldos = (Property::LDOS&)property;
	vector<double> &data = ldos.getDataRW();

	Property::GreensFunction greensFunction = pe->calculateGreensFunction(
		index,
		index,
		Property::GreensFunction::Type::NonPrincipal
	);
	const std::vector<complex<double>> &greensFunctionData
		= greensFunction.getData();

	int energyResolution = pe->getEnergyResolution();

	for(int n = 0; n < energyResolution; n++)
		data[offset + n] += imag(greensFunctionData[n])/M_PI;
}

void ChebyshevExpander::calculateSP_LDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	ChebyshevExpander *pe = (ChebyshevExpander*)cb_this;
	Property::SpinPolarizedLDOS &spinPolarizedLDOS
		= (Property::SpinPolarizedLDOS&)property;
	vector<SpinMatrix> &data = spinPolarizedLDOS.getDataRW();

	int spinIndex = information.getSpinIndex();
	Index to(index);
	Index from(index);

	int energyResolution = pe->getEnergyResolution();

	for(int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;		//up, up, down, down
		from.at(spinIndex) = n%2;	//up, down, up, down
		Property::GreensFunction greensFunction = pe->calculateGreensFunction(
			to,
			from,
			Property::GreensFunction::Type::NonPrincipal
		);
		const std::vector<complex<double>> &greensFunctionData
			= greensFunction.getData();

		for(int e = 0; e < energyResolution; e++)
			data[offset + e].at(n/2, n%2) += -i*greensFunctionData[e]/M_PI;
	}
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
