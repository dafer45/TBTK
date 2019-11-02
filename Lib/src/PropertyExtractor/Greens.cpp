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
#include "TBTK/PropertyExtractor/PatternValidator.h"
#include "TBTK/Functions.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <set>

using namespace std;

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

Property::Density Greens::calculateDensity(
	vector<Index> patterns
){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(1);
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	patternValidator.setCallingFunctionName(
		"PropertyExtractor::Greens::calculateDensity()"
	);
	patternValidator.validate(patterns);

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

Property::LDOS Greens::calculateLDOS(
	vector<Index> patterns
){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(1);
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	patternValidator.setCallingFunctionName(
		"PropertyExtractor::Greens::calculateLDOS()"
	);
	patternValidator.validate(patterns);

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

void Greens::calculateDensityCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	Greens *propertyExtractor = (Greens*)cb_this;
	Property::Density &density = (Property::Density&)property;
	vector<double> &data = density.getDataRW();

	const Property::GreensFunction &greensFunction
		= propertyExtractor->solver->getGreensFunction();

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

			data[offset] += -weight*imag(
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

			data[offset] += weight*imag(
				greensFunction({index, index}, e)
			)/M_PI*dE;
		}

		break;
	}
	case Property::GreensFunction::Type::Matsubara:
	{
		for(
			unsigned int e = 0;
			e < greensFunction.getNumMatsubaraEnergies();
			e++
		){
			data[offset] += real(greensFunction({index, index}, e));
		}

		data[offset]
			*= greensFunction.getFundamentalMatsubaraEnergy()/M_PI;

		data[offset] += 1/2.;

		break;
	}
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

void Greens::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	Greens *propertyExtractor = (Greens*)cb_this;
	Property::LDOS &ldos = (Property::LDOS&)property;
	vector<double> &data = ldos.getDataRW();

	const Property::GreensFunction &greensFunction
		= propertyExtractor->solver->getGreensFunction();

	switch(greensFunction.getType()){
	case Property::GreensFunction::Type::Retarded:
	{
		double lowerBound = greensFunction.getLowerBound();
		double upperBound = greensFunction.getUpperBound();
		int energyResolution = greensFunction.getResolution();

		const double dE = (upperBound - lowerBound)/energyResolution;
		for(int n = 0; n < energyResolution; n++){
			data[offset + n] -= imag(
				greensFunction({index, index}, n)
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
		for(int n = 0; n < energyResolution; n++){
			data[offset + n] += imag(
				greensFunction({index, index}, n)
			)/M_PI*dE;
		}

		break;
	}
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

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
