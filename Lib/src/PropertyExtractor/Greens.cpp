/* Copyright 2018 Kristofer Björnson
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

/** @file Greens.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PropertyExtractor/Greens.h"
#include "TBTK/Functions.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <set>

using namespace std;

static complex<double> i(0, 1);

namespace TBTK{
namespace PropertyExtractor{

Greens::Greens(Solver::Greens &solver){
	this->solver = &solver;
}

Greens::~Greens(){
}

void Greens::setEnergyWindow(
	double lowerBound,
	double upperBound,
	int energyResolution
){
	TBTKExit(
		"PropertyExtractor::Greens::setEnergyWindow()",
		"This function is not supported by this PropertyExtractor.",
		"The energy window is instead determined by the Green's"
		<< " function that is used by the corresponding solver. Use"
		<< " Solver::Greens::setGreensFunction() to set this Green's"
		<< " function."
	);
}

Property::Density Greens::calculateDensity(
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
		(void*)density.getDataRW().data(),
		pattern,
		ranges,
		0,
		1
	);

	return density;
}

Property::Density Greens::calculateDensity(
//	std::initializer_list<Index> patterns
	vector<Index> patterns
){
	validatePatternsNumComponents(
		patterns,
		1,
		"PropertyExtractor::Greens::calculateDensity()"
	);
	validatePatternsSpecifiers(
		patterns,
		{IDX_ALL, IDX_SUM_ALL},
		"PropertyExtractor::Greens::calculateDensity()"
	);

	IndexTree allIndices = generateIndexTree(
		patterns,
		solver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		solver->getModel().getHoppingAmplitudeSet(),
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

/*Property::Magnetization Greens::calculateMagnetization(
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
		calculateMagnetizationCallback,
		(void*)magnetization.getDataRW().data(),
		pattern,
		ranges,
		0,
		1
	);

	delete [] (int*)hint;

	return magnetization;
}

Property::Magnetization Greens::calculateMagnetization(
	std::initializer_list<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		solver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		solver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::Magnetization magnetization(memoryLayout);

	hint = new int[1];
	calculate(
		calculateMagnetizationCallback,
		allIndices,
		memoryLayout,
		magnetization,
		(int*)hint
	);

	delete [] (int*)hint;

	return magnetization;
}

Property::LDOS Greens::calculateLDOS(Index pattern, Index ranges){
	ensureCompliantRanges(pattern, ranges);

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

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
		(void*)ldos.getDataRW().data(),
		pattern,
		ranges,
		0,
		energyResolution
	);

	return ldos;
}*/

Property::LDOS Greens::calculateLDOS(
//	std::initializer_list<Index> patterns
	vector<Index> patterns
){
	validatePatternsNumComponents(
		patterns,
		1,
		"PropertyExtractor::Greens::calculateLDOS()"
	);
	validatePatternsSpecifiers(
		patterns,
		{IDX_ALL, IDX_SUM_ALL},
		"PropertyExtractor::Greens::calculateLDOS()"
	);

	IndexTree allIndices = generateIndexTree(
		patterns,
		solver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		solver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::LDOS ldos(
		memoryLayout,
		solver->getGreensFunction().getLowerBound(),
		solver->getGreensFunction().getUpperBound(),
		solver->getGreensFunction().getResolution()
	);

	calculate(
		calculateLDOSCallback,
		allIndices,
		memoryLayout,
		ldos
	);

	return ldos;
}

/*Property::SpinPolarizedLDOS Greens::calculateSpinPolarizedLDOS(
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
		getLowerBound(),
		getUpperBound(),
		getEnergyResolution()
	);

	calculate(
		calculateSpinPolarizedLDOSCallback,
		(void*)spinPolarizedLDOS.getDataRW().data(),
		pattern,
		ranges,
		0,
		getEnergyResolution()
	);

	delete [] (int*)hint;

	return spinPolarizedLDOS;
}

Property::SpinPolarizedLDOS Greens::calculateSpinPolarizedLDOS(
	std::initializer_list<Index> patterns
){
	hint = new int[1];

	IndexTree allIndices = generateIndexTree(
		patterns,
		solver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		solver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		memoryLayout,
		getLowerBound(),
		getUpperBound(),
		getEnergyResolution()
	);

	calculate(
		calculateSpinPolarizedLDOSCallback,
		allIndices,
		memoryLayout,
		spinPolarizedLDOS,
		(int*)hint
	);

	delete [] (int*)hint;

	return spinPolarizedLDOS;
}*/

void Greens::calculateDensityCallback(
	PropertyExtractor *cb_this,
	void *density,
	const Index &index,
	int offset
){
	Greens *propertyExtractor = (Greens*)cb_this;

	const Property::GreensFunction &greensFunction
		= propertyExtractor->solver->getGreensFunction();
/*	calculateGreensFunction(
		index,
		index,
		Property::GreensFunction::Type::NonPrincipal
	);
	const std::vector<complex<double>> &greensFunctionData
		= greensFunction.getData();*/

	Statistics statistics = propertyExtractor->solver->getModel().getStatistics();

	switch(greensFunction.getType()){
	case Property::GreensFunction::Type::Retarded:
	{
		double lowerBound = greensFunction.getLowerBound();
		double upperBound = greensFunction.getUpperBound();
		int energyResolution = greensFunction.getResolution();

		const double dE = (upperBound - lowerBound)/energyResolution;
		for(int e = 0; e < energyResolution; e++){
			double E = lowerBound + e*dE;

			double weight;
			if(statistics == Statistics::FermiDirac){
				weight = Functions::fermiDiracDistribution(
					E,
					propertyExtractor->solver->getModel(
					).getChemicalPotential(),
					propertyExtractor->solver->getModel(
					).getTemperature()
				);
			}
			else{
				weight = Functions::boseEinsteinDistribution(
					E,
					propertyExtractor->solver->getModel(
					).getChemicalPotential(),
					propertyExtractor->solver->getModel().getTemperature()
				);
			}

			((double*)density)[offset] += -weight*imag(
				greensFunction({index, index}, e)
			)/M_PI*dE;
		}

		break;
	}
	case Property::GreensFunction::Type::Advanced:
	case Property::GreensFunction::Type::NonPrincipal:
	{
		double lowerBound = greensFunction.getLowerBound();
		double upperBound = greensFunction.getUpperBound();
		int energyResolution = greensFunction.getResolution();

		const double dE = (upperBound - lowerBound)/energyResolution;
		for(int e = 0; e < energyResolution; e++){
			double E = lowerBound + e*dE;

			double weight;
			if(statistics == Statistics::FermiDirac){
				weight = Functions::fermiDiracDistribution(
					E,
					propertyExtractor->solver->getModel(
					).getChemicalPotential(),
					propertyExtractor->solver->getModel().getTemperature()
				);
			}
			else{
				weight = Functions::boseEinsteinDistribution(
					E,
					propertyExtractor->solver->getModel(
					).getChemicalPotential(),
					propertyExtractor->solver->getModel().getTemperature()
				);
			}

			((double*)density)[offset] += weight*imag(
				greensFunction({index, index}, e)
			)/M_PI*dE;
		}

		break;
	}
	case Property::GreensFunction::Type::Principal:
	default:
		TBTKExit(
			"PropertyExtractor::Greens::calculateDensityCallback()",
			"Only calculation of the Density from the Retarded,"
			<< " Advanced, and NonPrincipal Green's function"
			<< " is supported yet.",
			""
		);
	}
}

/*void ChebyshevExpander::calculateMAGCallback(
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

			((SpinMatrix*)mag)[offset].at(n/2, n%2) += weight*(-i)*greensFunctionData[e]/M_PI*dE;
		}
	}
}*/

void Greens::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	void *ldos,
	const Index &index,
	int offset
){
	Greens *propertyExtractor = (Greens*)cb_this;

	const Property::GreensFunction &greensFunction
		= propertyExtractor->solver->getGreensFunction();
/*	Property::GreensFunction greensFunction = pe->calculateGreensFunction(
		index,
		index,
		Property::GreensFunction::Type::NonPrincipal
	);
	const std::vector<complex<double>> &greensFunctionData
		= greensFunction.getData();*/

	switch(greensFunction.getType()){
	case Property::GreensFunction::Type::Retarded:
	{
		double lowerBound = greensFunction.getLowerBound();
		double upperBound = greensFunction.getUpperBound();
		int energyResolution = greensFunction.getResolution();

		const double dE = (upperBound - lowerBound)/energyResolution;
		for(int n = 0; n < energyResolution; n++)
			((double*)ldos)[offset + n] -= imag(
				greensFunction({index, index}, n)
			)/M_PI*dE;

		break;
	}
	case Property::GreensFunction::Type::Advanced:
	case Property::GreensFunction::Type::NonPrincipal:
	{
		double lowerBound = greensFunction.getLowerBound();
		double upperBound = greensFunction.getUpperBound();
		int energyResolution = greensFunction.getResolution();

		const double dE = (upperBound - lowerBound)/energyResolution;
		for(int n = 0; n < energyResolution; n++)
			((double*)ldos)[offset + n] += imag(
				greensFunction({index, index}, n)
			)/M_PI*dE;

		break;
	}
	case Property::GreensFunction::Type::Principal:
	default:
		TBTKExit(
			"PropertyExtractor::Greens::calculateDensityCallback()",
			"Only calculation of the Density from the Retarded,"
			<< " Advanced, and NonPrincipal Green's function"
			<< " is supported yet.",
			""
		);
	}
}

/*void ChebyshevExpander::calculateSP_LDOSCallback(
	PropertyExtractor *cb_this,
	void *sp_ldos,
	const Index &index,
	int offset
){
	ChebyshevExpander *pe = (ChebyshevExpander*)cb_this;

	int spinIndex = ((int*)(pe->hint))[0];
	Index to(index);
	Index from(index);

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();

	const double dE = (upperBound - lowerBound)/energyResolution;
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
			((SpinMatrix*)sp_ldos)[offset + e].at(n/2, n%2) += -i*greensFunctionData[e]/M_PI*dE;
	}
}*/

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
