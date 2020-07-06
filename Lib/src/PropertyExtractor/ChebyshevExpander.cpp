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

ChebyshevExpander::ChebyshevExpander(Solver::ChebyshevExpander &solver){
	const Range &energyWindow = getEnergyWindow();
	TBTKAssert(
		energyWindow.getResolution() > 0,
		"PropertyExtractor::ChebyshevExpander::ChebyshevExpander()",
		"The energy resolution has to be a positive number.",
		""
	);
	TBTKAssert(
		energyWindow[0] < energyWindow.getLast(),
		"PropertyExtractor::ChebyshevExpander::ChebyshevExpander()",
		"The energy window must be accending.",
		""
	);
	TBTKAssert(
		energyWindow[0] >= -solver.getScaleFactor(),
		"PropertyExtractor::ChebyshevExpander::ChebyshevExpander()",
		"The lower bound for the energy window has to be larger than"
		<< " -solver->getScaleFactor().",
		"Use Solver::ChebyshevExpander::setScaleFactor() to set a"
		<< " larger scale factor."
	);
	TBTKAssert(
		energyWindow.getLast() <= solver.getScaleFactor(),
		"PropertyExtractor::ChebyshevExapnder::ChebysheExpander()",
		"The upper bound for the energy window has to be smaller than"
		<< " solver->getScaleFactor().",
		"Use Solver::ChebyshevExpnader::setScaleFactor() to set a"
		<< " larger scale factor."
	);

	setSolver(solver);

	setEnergyWindow(
		-solver.getScaleFactor(),
		solver.getScaleFactor(),
		energyWindow.getResolution()
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

	Solver::ChebyshevExpander &solver = getSolver();
	solver.setEnergyWindow(
		Range(lowerBound, upperBound, energyResolution)
	);
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
	const Solver::ChebyshevExpander &solver = getSolver();
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
			solver.getModel().getHoppingAmplitudeSet(),
			true,
			true
		);
		IndexTree fromTree = generateIndexTree(
			{fromPattern},
			solver.getModel().getHoppingAmplitudeSet(),
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
		getEnergyWindow()
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

			for(
				unsigned int c = 0;
				c < getEnergyWindow().getResolution();
				c++
			){
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
	Solver::ChebyshevExpander &solver = getSolver();
	vector<vector<complex<double>>> coefficients = solver.calculateCoefficients(
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
		getEnergyWindow()
	);
	std::vector<complex<double>> &data = greensFunction.getDataRW();

	#pragma omp parallel for
	for(unsigned int n = 0; n < to.size(); n++){
		vector<complex<double>> greensFunctionData = solver.generateGreensFunction(
			coefficients[n],
			chebyshevType
		);
		unsigned int offset = greensFunction.getOffset({to[n], from});
		for(
			unsigned int c = 0;
			c < getEnergyWindow().getResolution();
			c++
		){
			data[offset + c] = greensFunctionData[c];
		}
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

	const Model& model = getSolver().getModel();

	const Range &energyWindow = getEnergyWindow();
	const double dE = (energyWindow[1] - energyWindow[0]);
	for(unsigned int e = 0; e < energyWindow.getResolution(); e++){
		double weight = getThermodynamicEquilibriumOccupation(
			energyWindow[e],
			model
		);

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
	const Solver::ChebyshevExpander &solver = getSolver();
	IndexTree allIndices = generateIndexTree(
		patterns,
		solver.getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		solver.getModel().getHoppingAmplitudeSet(),
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
	const Solver::ChebyshevExpander &solver = getSolver();
	IndexTree allIndices = generateIndexTree(
		patterns,
		solver.getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		solver.getModel().getHoppingAmplitudeSet(),
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

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::LDOS ldos(loopRanges, getEnergyWindow());

	Information information;
	calculate(
		calculateLDOSCallback,
		ldos,
		pattern,
		ranges,
		0,
		getEnergyWindow().getResolution(),
		information
	);

	return ldos;
}

Property::LDOS ChebyshevExpander::calculateLDOS(
	vector<Index> patterns
){
	const Solver::ChebyshevExpander &solver = getSolver();
	IndexTree allIndices = generateIndexTree(
		patterns,
		solver.getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		solver.getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::LDOS ldos(memoryLayout, getEnergyWindow());

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
		getEnergyWindow()
	);

	calculate(
		calculateSP_LDOSCallback,
		spinPolarizedLDOS,
		pattern,
		ranges,
		0,
		getEnergyWindow().getResolution(),
		information
	);

	return spinPolarizedLDOS;
}

Property::SpinPolarizedLDOS ChebyshevExpander::calculateSpinPolarizedLDOS(
	vector<Index> patterns
){
	const Solver::ChebyshevExpander &solver = getSolver();
	IndexTree allIndices = generateIndexTree(
		patterns,
		solver.getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		solver.getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		memoryLayout,
		getEnergyWindow()
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
	ChebyshevExpander *propertyExtractor = (ChebyshevExpander*)cb_this;
	Property::Density &density = (Property::Density&)property;
	vector<double> &data = density.getDataRW();

	Property::GreensFunction greensFunction
		= propertyExtractor->calculateGreensFunction(
			index,
			index,
			Property::GreensFunction::Type::NonPrincipal
		);
	const std::vector<complex<double>> &greensFunctionData
		= greensFunction.getData();

	const Model &model = propertyExtractor->getSolver().getModel();

	const Range &energyWindow = propertyExtractor->getEnergyWindow();
	const double dE = energyWindow[1] - energyWindow[0];
	for(unsigned int e = 0; e < energyWindow.getResolution(); e++){
		double weight = getThermodynamicEquilibriumOccupation(
			energyWindow[e],
			model
		);

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
	ChebyshevExpander *propertyExtractor = (ChebyshevExpander*)cb_this;
	const Model &model = propertyExtractor->getSolver().getModel();
	Property::Magnetization &magnetization
		= (Property::Magnetization&)property;
	vector<SpinMatrix> &data = magnetization.getDataRW();

	int spinIndex = information.getSpinIndex();
	Index to(index);
	Index from(index);

	const Range &energyWindow = propertyExtractor->getEnergyWindow();
	const double dE = energyWindow[1] - energyWindow[0];
	for(int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;		//up, up, down, down
		from.at(spinIndex) = n%2;	//up, down, up, down
		Property::GreensFunction greensFunction
			= propertyExtractor->calculateGreensFunction(
				to,
				from,
				Property::GreensFunction::Type::NonPrincipal
			);
		const std::vector<complex<double>> &greensFunctionData
			= greensFunction.getData();

		for(unsigned int e = 0; e < energyWindow.getResolution(); e++){
			double weight = getThermodynamicEquilibriumOccupation(
				energyWindow[e],
				model
			);

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
	ChebyshevExpander *propertyExtractor = (ChebyshevExpander*)cb_this;
	Property::LDOS &ldos = (Property::LDOS&)property;
	vector<double> &data = ldos.getDataRW();

	Property::GreensFunction greensFunction
		= propertyExtractor->calculateGreensFunction(
			index,
			index,
			Property::GreensFunction::Type::NonPrincipal
		);
	const std::vector<complex<double>> &greensFunctionData
		= greensFunction.getData();

	const Range &energyWindow = propertyExtractor->getEnergyWindow();
	const double dE = energyWindow[1] - energyWindow[0];
	for(unsigned int n = 0; n < energyWindow.getResolution(); n++)
		data[offset + n] += imag(greensFunctionData[n])/M_PI*dE;
}

void ChebyshevExpander::calculateSP_LDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	ChebyshevExpander *propertyExtractor = (ChebyshevExpander*)cb_this;
	Property::SpinPolarizedLDOS &spinPolarizedLDOS
		= (Property::SpinPolarizedLDOS&)property;
	vector<SpinMatrix> &data = spinPolarizedLDOS.getDataRW();

	int spinIndex = information.getSpinIndex();
	Index to(index);
	Index from(index);

	const Range &energyWindow = propertyExtractor->getEnergyWindow();
	const double dE = energyWindow[1] - energyWindow[0];
	for(int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;		//up, up, down, down
		from.at(spinIndex) = n%2;	//up, down, up, down
		Property::GreensFunction greensFunction
			= propertyExtractor->calculateGreensFunction(
				to,
				from,
				Property::GreensFunction::Type::NonPrincipal
			);
		const std::vector<complex<double>> &greensFunctionData
			= greensFunction.getData();

		for(unsigned int e = 0; e < energyWindow.getResolution(); e++){
			data[offset + e].at(n/2, n%2)
				+= -i*greensFunctionData[e]/M_PI*dE;
		}
	}
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
