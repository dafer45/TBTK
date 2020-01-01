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

/** @file Diagonalizer.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/PropertyExtractor/IndexTreeGenerator.h"
#include "TBTK/PropertyExtractor/PatternValidator.h"
#include "TBTK/Functions.h"
#include "TBTK/Streams.h"

#include <cmath>

using namespace std;

static complex<double> i(0, 1);

namespace TBTK{
namespace PropertyExtractor{

Diagonalizer::Diagonalizer(Solver::Diagonalizer &solver) : solver(solver){
}

Property::EigenValues Diagonalizer::getEigenValues(){
	const CArray<double> &eigenValues = solver.getEigenValues();

	return Property::EigenValues(
		eigenValues.getSize(),
		eigenValues.getData()
	);
}

Property::WaveFunctions Diagonalizer::calculateWaveFunctions(
	vector<Index> patterns,
	vector<Subindex> states
){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(1);
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	patternValidator.setCallingFunctionName(
		"PropertyExtractor::Diagonalizer::calculateWaveFunction()"
	);
	patternValidator.validate(patterns);

	IndexTreeGenerator indexTreeGenerator(solver.getModel());
	IndexTree allIndices = indexTreeGenerator.generateAllIndices(patterns);
	IndexTree memoryLayout
		= indexTreeGenerator.generateMemoryLayout(patterns);

	vector<unsigned int> statesVector;
	if(states.size() == 1){
		if((*states.begin()).isWildcard()){
			for(int n = 0; n < solver.getModel().getBasisSize(); n++)
				statesVector.push_back(n);
		}
		else{
			TBTKAssert(
				*states.begin() >= 0,
				"PropertyExtractor::Diagonalizer::calculateWaveFunctions()",
				"Found unexpected index symbol.",
				"Use only positive numbers or '{IDX_ALL}'"
			);
			statesVector.push_back(*states.begin());
		}
	}
	else{
		for(unsigned int n = 0; n < states.size(); n++){
			TBTKAssert(
				*(states.begin() + n) >= 0,
				"PropertyExtractor::Diagonalizer::calculateWaveFunctions()",
				"Found unexpected index symbol.",
				"Use only positive numbers or '{IDX_ALL}'"
			);
			statesVector.push_back(*(states.begin() + n));
		}
	}

	Property::WaveFunctions waveFunctions(memoryLayout, statesVector);

	Information information;
	calculate(
		calculateWaveFunctionsCallback,
		allIndices,
		memoryLayout,
		waveFunctions,
		information
	);

	return waveFunctions;
}

Property::GreensFunction Diagonalizer::calculateGreensFunction(
	const vector<Index> &patterns,
	Property::GreensFunction::Type type
){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(2);
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	patternValidator.setCallingFunctionName(
		"PropertyExtractor::Diagonalizer::calculateGreensFunction()"
	);
	patternValidator.validate(patterns);

	IndexTreeGenerator indexTreeGenerator(solver.getModel());
	IndexTree allIndices = indexTreeGenerator.generateAllIndices(patterns);
	IndexTree memoryLayout
		= indexTreeGenerator.generateMemoryLayout(patterns);

	Property::GreensFunction greensFunction;

	switch(type){
	case Property::GreensFunction::Type::Advanced:
	case Property::GreensFunction::Type::Retarded:
	{
		greensFunction = Property::GreensFunction(
			memoryLayout,
			type,
			getLowerBound(),
			getUpperBound(),
			getEnergyResolution()
		);

		Information information;
		calculate(
			calculateGreensFunctionCallback,
			allIndices,
			memoryLayout,
			greensFunction,
			information
		);

		break;
	}
	case Property::GreensFunction::Type::Matsubara:
	{
		int lowerFermionicMatsubaraEnergyIndex
			= getLowerFermionicMatsubaraEnergyIndex();
		int upperFermionicMatsubaraEnergyIndex
			= getUpperFermionicMatsubaraEnergyIndex();

		TBTKAssert(
			lowerFermionicMatsubaraEnergyIndex
				<= upperFermionicMatsubaraEnergyIndex,
			"PropertyExtractor::Diagonalizer::caluclateGreensFunction()",
			"'lowerFermionicMatsubaraEnergyIndex="
			<< lowerFermionicMatsubaraEnergyIndex << "' must be"
			<< " less or equal to"
			<< " 'upperFermionicMatsubaraEnergyIndex="
			<< upperFermionicMatsubaraEnergyIndex << "'.",
			"This should never happen, contact the developer."
		);

		double temperature = solver.getModel().getTemperature();
		double kT = UnitHandler::getConstantInNaturalUnits("k_B")*temperature;
		double fundamentalMatsubaraEnergy = M_PI*kT;

		greensFunction = Property::GreensFunction(
			memoryLayout,
			lowerFermionicMatsubaraEnergyIndex,
			upperFermionicMatsubaraEnergyIndex,
			fundamentalMatsubaraEnergy
		);

		Information information;
		calculate(
			calculateGreensFunctionCallback,
			allIndices,
			memoryLayout,
			greensFunction,
			information
		);

		break;
	}
	default:
		TBTKExit(
			"PropertyExtractor::Diagonalizer::calculateGreensFunction()",
			"Only type Property::GreensFunction::Type::Advanced,"
			<< " Property::GrensFunction::Type::Retarded, and"
			<< " Property::GreensFunction::Type::Matsubara"
			<< " supported yet.",
			""
		);
	}

	return greensFunction;
}

Property::DOS Diagonalizer::calculateDOS(){
	const CArray<double> &eigenValues = solver.getEigenValues();
	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	Property::DOS dos(lowerBound, upperBound, energyResolution);
	double dE = dos.getDeltaE();
	for(unsigned int n = 0; n < eigenValues.getSize(); n++){
		int e = round((eigenValues[n] - lowerBound)/dE);
		if(e >= 0 && e < energyResolution)
			dos(e) += 1./dE;
	}

	return dos;
}

complex<double> Diagonalizer::calculateExpectationValue(
	Index to,
	Index from
){
	const complex<double> i(0, 1);

	complex<double> expectationValue = 0.;

	Statistics statistics = solver.getModel().getStatistics();

	for(int n = 0; n < solver.getModel().getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				solver.getEigenValue(n),
				solver.getModel().getChemicalPotential(),
				solver.getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				solver.getEigenValue(n),
				solver.getModel().getChemicalPotential(),
				solver.getModel().getTemperature()
			);
		}

		complex<double> u_to = solver.getAmplitude(n, to);
		complex<double> u_from = solver.getAmplitude(n, from);

		expectationValue += weight*conj(u_to)*u_from;
	}

	return expectationValue;
}

Property::Density Diagonalizer::calculateDensity(
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

Property::Density Diagonalizer::calculateDensity(
	vector<Index> patterns
){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(1);
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	patternValidator.setCallingFunctionName(
		"PropertyExtractor::Diagonalizer::calculateDensity()"
	);
	patternValidator.validate(patterns);

	IndexTreeGenerator indexTreeGenerator(solver.getModel());
	IndexTree allIndices = indexTreeGenerator.generateAllIndices(patterns);
	IndexTree memoryLayout
		= indexTreeGenerator.generateMemoryLayout(patterns);

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

Property::Magnetization Diagonalizer::calculateMagnetization(
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
			"PropertyExtractor::Diagonalizer::calculateMagnetization()",
			"No spin index indiceated.",
			"Used IDX_SPIN to indicate the position of the spin index."
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

Property::Magnetization Diagonalizer::calculateMagnetization(
	vector<Index> patterns
){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(1);
	patternValidator.setAllowedSubindexFlags(
		{IDX_ALL, IDX_SUM_ALL, IDX_SPIN}
	);
	patternValidator.setCallingFunctionName(
		"PropertyExtractor::Diagonalizer::calculateMagnetization()"
	);
	patternValidator.validate(patterns);

	IndexTreeGenerator indexTreeGenerator(solver.getModel());
	IndexTree allIndices = indexTreeGenerator.generateAllIndices(patterns);
	IndexTree memoryLayout
		= indexTreeGenerator.generateMemoryLayout(patterns);

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

Property::LDOS Diagonalizer::calculateLDOS(
	Index pattern,
	Index ranges
){
	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	ensureCompliantRanges(pattern, ranges);

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

Property::LDOS Diagonalizer::calculateLDOS(
	vector<Index> patterns
){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(1);
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	patternValidator.setCallingFunctionName(
		"PropertyExtractor::Diagonalizer::calculateLDOS()"
	);
	patternValidator.validate(patterns);

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	IndexTreeGenerator indexTreeGenerator(solver.getModel());
	IndexTree allIndices = indexTreeGenerator.generateAllIndices(patterns);
	IndexTree memoryLayout
		= indexTreeGenerator.generateMemoryLayout(patterns);

	Property::LDOS ldos(
		memoryLayout,
		lowerBound,
		upperBound,
		energyResolution
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

Property::SpinPolarizedLDOS Diagonalizer::calculateSpinPolarizedLDOS(
	Index pattern,
	Index ranges
){
	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

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
			"PropertyExtractor::Diagonalizer::calculateSpinPolarizedLDOS()",
			"No spin index indicated.",
			"Used IDX_SPIN to indicate the position of the spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		loopRanges,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateSP_LDOSCallback,
		spinPolarizedLDOS,
		pattern,
		ranges,
		0,
		energyResolution,
		information
	);

	return spinPolarizedLDOS;
}

Property::SpinPolarizedLDOS Diagonalizer::calculateSpinPolarizedLDOS(
	vector<Index> patterns
){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(1);
	patternValidator.setAllowedSubindexFlags(
		{IDX_ALL, IDX_SUM_ALL, IDX_SPIN}
	);
	patternValidator.setCallingFunctionName(
		"PropertyExtractor::Diagonalizer::calculateSpinPolarizedLDOS()"
	);
	patternValidator.validate(patterns);

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	IndexTreeGenerator indexTreeGenerator(solver.getModel());
	IndexTree allIndices = indexTreeGenerator.generateAllIndices(patterns);
	IndexTree memoryLayout
		= indexTreeGenerator.generateMemoryLayout(patterns);

	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		memoryLayout,
		lowerBound,
		upperBound,
		energyResolution
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

double Diagonalizer::calculateEntropy(){
	Statistics statistics = solver.getModel().getStatistics();

	double entropy = 0.;
	for(int n = 0; n < solver.getModel().getBasisSize(); n++){
		double p;

		switch(statistics){
		case Statistics::FermiDirac:
			p = Functions::fermiDiracDistribution(
				getEigenValue(n),
				solver.getModel().getChemicalPotential(),
				solver.getModel().getTemperature()
			);
			break;
		case Statistics::BoseEinstein:
			p = Functions::boseEinsteinDistribution(
				getEigenValue(n),
				solver.getModel().getChemicalPotential(),
				solver.getModel().getTemperature()
			);
			break;
		default:
			TBTKExit(
				"PropertyExtractor::Diagonalizer::calculateEntropy()",
				"Unknown statistics.",
				"This should never happen, contact the developer."
			);
		}

		entropy -= p*log(p);
	}

	entropy *= UnitHandler::getConstantInNaturalUnits("k_B");

	return entropy;
}

void Diagonalizer::calculateWaveFunctionsCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	Diagonalizer *propertyExtractor = (Diagonalizer*)cb_this;
	Property::WaveFunctions &waveFunctions
		= (Property::WaveFunctions&)property;
	vector<complex<double>> &data = waveFunctions.getDataRW();

	const vector<unsigned int> states = waveFunctions.getStates();
	for(unsigned int n = 0; n < states.size(); n++)
		data[offset + n] += propertyExtractor->getAmplitude(states.at(n), index);
}

void Diagonalizer::calculateGreensFunctionCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	Diagonalizer *propertyExtractor = (Diagonalizer*)cb_this;

	vector<Index> components = index.split();

	Property::GreensFunction &greensFunction
		= (Property::GreensFunction&)property;
	vector<complex<double>> &data = greensFunction.getDataRW();

	switch(greensFunction.getType()){
	case Property::GreensFunction::Type::Advanced:
	case Property::GreensFunction::Type::Retarded:
	{
		double lowerBound = propertyExtractor->getLowerBound();
		int energyResolution = propertyExtractor->getEnergyResolution();
		double dE = greensFunction.getDeltaE();
		double delta;
		switch(greensFunction.getType()){
			case Property::GreensFunction::Type::Advanced:
				delta = -propertyExtractor->getEnergyInfinitesimal();
				break;
			case Property::GreensFunction::Type::Retarded:
				delta = propertyExtractor->getEnergyInfinitesimal();
				break;
			default:
				TBTKExit(
					"Diagonalizer::calculateGreensFunctionCallback()",
					"Unknown Green's function type.",
					"This should never happen, contact the developer."
				);
		}

		for(int e = 0; e < energyResolution; e++){
			double E = lowerBound + e*dE;

			for(
				int n = 0;
				n < propertyExtractor->solver.getModel().getBasisSize();
				n++
			){
				double E_n = propertyExtractor->getEigenValue(n);
				complex<double> amplitude0
					= propertyExtractor->getAmplitude(n, components[0]);
				complex<double> amplitude1
					= propertyExtractor->getAmplitude(n, components[1]);
				data[offset + e]
					+= amplitude0*conj(amplitude1)/(
						E - E_n + i*delta
					);
			}
		}

		break;
	}
	case Property::GreensFunction::Type::Matsubara:
	{
		unsigned int numMatsubaraEnergies
			= greensFunction.getNumMatsubaraEnergies();
		double chemicalPotential = propertyExtractor->solver.getModel(
			).getChemicalPotential();

		for(unsigned int e = 0; e < numMatsubaraEnergies; e++){
			complex<double> E = greensFunction.getMatsubaraEnergy(e)
				+ chemicalPotential;

			for(
				int n = 0;
				n < propertyExtractor->solver.getModel().getBasisSize();
				n++
			){
				double E_n = propertyExtractor->getEigenValue(n);
				complex<double> amplitude0
					= propertyExtractor->getAmplitude(n, components[0]);
				complex<double> amplitude1
					= propertyExtractor->getAmplitude(n, components[1]);
				data[offset + e]
					+= amplitude0*conj(amplitude1)/(
						E - E_n
					);
			}
		}

		break;
	}
	default:
		TBTKExit(
			"Diagonalizer::calculateGreensFunctionCallback()",
			"Only type Property::GreensFunction::Type::Advanced,"
			<< " Property::GreensFunction::Type::Retarded, and"
			<< " Property::GreensFunction::Type::Matsubara"
			<< " supported yet.",
			""
		);
	}
}

void Diagonalizer::calculateDensityCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	Diagonalizer *propertyExtractor = (Diagonalizer*)cb_this;
	Property::Density &density = (Property::Density&)property;
	vector<double> &data = density.getDataRW();
	Solver::Diagonalizer &solver = propertyExtractor->solver;
	const Model &model = solver.getModel();

	const CArray<double> &eigenValues = solver.getEigenValues();
	Statistics statistics = model.getStatistics();
	for(int n = 0; n < model.getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				eigenValues[n],
				model.getChemicalPotential(),
				model.getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				eigenValues[n],
				model.getChemicalPotential(),
				model.getTemperature()
			);
		}

		complex<double> u = solver.getAmplitude(n, index);

		data[offset] += pow(abs(u), 2)*weight;
	}
}

void Diagonalizer::calculateMAGCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	Diagonalizer *propertyExtractor = (Diagonalizer*)cb_this;
	Property::Magnetization &magnetization
		= (Property::Magnetization&)property;
	vector<SpinMatrix> &data = magnetization.getDataRW();
	Solver::Diagonalizer &solver = propertyExtractor->solver;
	const Model &model = solver.getModel();

	const CArray<double> &eigenValues = solver.getEigenValues();
	Statistics statistics = model.getStatistics();

	int spinIndex = information.getSpinIndex();
	Index index_u(index);
	Index index_d(index);
	index_u.at(spinIndex) = 0;
	index_d.at(spinIndex) = 1;
	for(int n = 0; n < model.getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				eigenValues[n],
				model.getChemicalPotential(),
				model.getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				eigenValues[n],
				model.getChemicalPotential(),
				model.getTemperature()
			);
		}

		complex<double> u_u = solver.getAmplitude(n, index_u);
		complex<double> u_d = solver.getAmplitude(n, index_d);

		data[offset].at(0, 0) += conj(u_u)*u_u*weight;
		data[offset].at(0, 1) += conj(u_u)*u_d*weight;
		data[offset].at(1, 0) += conj(u_d)*u_u*weight;
		data[offset].at(1, 1) += conj(u_d)*u_d*weight;
	}
}

void Diagonalizer::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	Diagonalizer *propertyExtractor = (Diagonalizer*)cb_this;
	Property::LDOS &ldos = (Property::LDOS&)property;
	vector<double> &data = ldos.getDataRW();
	Solver::Diagonalizer &solver = propertyExtractor->solver;

	double lowerBound = propertyExtractor->getLowerBound();
	double upperBound = propertyExtractor->getUpperBound();
	int energyResolution = propertyExtractor->getEnergyResolution();

	const CArray<double> &eigenValues = solver.getEigenValues();

	double dE = ldos.getDeltaE();
	for(unsigned int n = 0; n < eigenValues.getSize(); n++){
		if(eigenValues[n] > lowerBound && eigenValues[n] < upperBound){
			complex<double> u = solver.getAmplitude(n, index);

			int e = round((eigenValues[n] - lowerBound)/dE);
			if(e >= 0 && e < energyResolution)
				data[offset + e] += real(conj(u)*u)/dE;
		}
	}
}

void Diagonalizer::calculateSP_LDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	Diagonalizer *propertyExtractor = (Diagonalizer*)cb_this;
	Property::SpinPolarizedLDOS &spinPolarizedLDOS
		= (Property::SpinPolarizedLDOS&)property;
	vector<SpinMatrix> &data = spinPolarizedLDOS.getDataRW();
	Solver::Diagonalizer &solver = propertyExtractor->solver;
	const Model &model = solver.getModel();

	double lowerBound = propertyExtractor->getLowerBound();
	double upperBound = propertyExtractor->getUpperBound();
	int energyResolution = propertyExtractor->getEnergyResolution();

	const CArray<double> &eigenValues = solver.getEigenValues();

	int spinIndex = information.getSpinIndex();

	Index index_u(index);
	Index index_d(index);
	index_u.at(spinIndex) = 0;
	index_d.at(spinIndex) = 1;
	double dE = spinPolarizedLDOS.getDeltaE();
	for(int n = 0; n < model.getBasisSize(); n++){
		if(eigenValues[n] > lowerBound && eigenValues[n] < upperBound){
			complex<double> u_u = solver.getAmplitude(n, index_u);
			complex<double> u_d = solver.getAmplitude(n, index_d);

			int e = (int)((eigenValues[n] - lowerBound)/dE);
			if(e >= energyResolution)
				e = energyResolution - 1;
			data[offset + e].at(0, 0) += conj(u_u)*u_u/dE;
			data[offset + e].at(0, 1) += conj(u_u)*u_d/dE;
			data[offset + e].at(1, 0) += conj(u_d)*u_u/dE;
			data[offset + e].at(1, 1) += conj(u_d)*u_d/dE;
		}
	}
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
