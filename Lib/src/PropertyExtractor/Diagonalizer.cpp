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

Diagonalizer::Diagonalizer(){
}

Property::EigenValues Diagonalizer::getEigenValues(){
	const CArray<double> &eigenValues = getSolver().getEigenValues();
	return Property::EigenValues(eigenValues.getSize(), eigenValues);
}

Property::WaveFunctions Diagonalizer::calculateWaveFunctions(
	vector<Index> patterns,
	vector<Subindex> states
){
	PatternValidator::validateWaveFunctionPatterns(patterns);
	IndexTree allIndices = generateAllIndices(patterns);
	IndexTree memoryLayout = generateMemoryLayout(patterns);
	const Model &model = getSolver().getModel();

	vector<unsigned int> statesVector;
	if(states.size() == 1){
		if((*states.begin()).isWildcard()){
			for(int n = 0; n < model.getBasisSize(); n++)
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
	PatternValidator::validateGreensFunctionPatterns(patterns);
	IndexTree allIndices = generateAllIndices(patterns);
	IndexTree memoryLayout = generateMemoryLayout(patterns);
	const Model &model = getSolver().getModel();

	Property::GreensFunction greensFunction;

	switch(type){
	case Property::GreensFunction::Type::Advanced:
	case Property::GreensFunction::Type::Retarded:
	{
		greensFunction = Property::GreensFunction(
			memoryLayout,
			type,
			getEnergyWindow()
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

		double temperature = model.getTemperature();
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
	const CArray<double> &eigenValues = getSolver().getEigenValues();
	const Range &energyWindow = getEnergyWindow();
	Property::DOS dos(energyWindow);
	double dE = dos.getDeltaE();
	for(unsigned int n = 0; n < eigenValues.getSize(); n++){
		int e = round((eigenValues[n] - energyWindow[0])/dE);
		if(e >= 0 && e < (int)energyWindow.getResolution())
			dos(e) += 1./dE;
	}

	return dos;
}

complex<double> Diagonalizer::calculateExpectationValue(
	Index to,
	Index from
){
	complex<double> expectationValue = 0.;
	const Model &model = getSolver().getModel();
	for(int n = 0; n < model.getBasisSize(); n++){
		double weight = getThermodynamicEquilibriumOccupation(
			getEigenValue(n),
			model
		);
		complex<double> amplitudeTo = getAmplitude(n, to);
		complex<double> amplitudeFrom = getAmplitude(n, from);

		expectationValue += weight*conj(amplitudeTo)*amplitudeFrom;
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

Property::Density Diagonalizer::calculateDensity(vector<Index> patterns){
	PatternValidator::validateDensityPatterns(patterns);
	IndexTree allIndices = generateAllIndices(patterns);
	IndexTree memoryLayout = generateMemoryLayout(patterns);

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
	PatternValidator::validateMagnetizationPatterns(patterns);
	IndexTree allIndices = generateAllIndices(patterns);
	IndexTree memoryLayout = generateMemoryLayout(patterns);

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

Property::LDOS Diagonalizer::calculateLDOS(
	vector<Index> patterns
){
	PatternValidator::validateLDOSPatterns(patterns);
	IndexTree allIndices = generateAllIndices(patterns);
	IndexTree memoryLayout = generateMemoryLayout(patterns);

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

Property::SpinPolarizedLDOS Diagonalizer::calculateSpinPolarizedLDOS(
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
			"PropertyExtractor::Diagonalizer::calculateSpinPolarizedLDOS()",
			"No spin index indicated.",
			"Used IDX_SPIN to indicate the position of the spin index."
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

Property::SpinPolarizedLDOS Diagonalizer::calculateSpinPolarizedLDOS(
	vector<Index> patterns
){
	PatternValidator::validateSpinPolarizedLDOSPatterns(patterns);
	IndexTree allIndices = generateAllIndices(patterns);
	IndexTree memoryLayout = generateMemoryLayout(patterns);

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

double Diagonalizer::calculateEntropy(){
	double entropy = 0.;
	const Model &model = getSolver().getModel();
	for(int n = 0; n < model.getBasisSize(); n++){
		double p = getThermodynamicEquilibriumOccupation(
			getEigenValue(n),
			model
		);
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
	const Model &model = propertyExtractor->getSolver().getModel();

	vector<Index> components = index.split();

	Property::GreensFunction &greensFunction
		= (Property::GreensFunction&)property;
	vector<complex<double>> &data = greensFunction.getDataRW();

	switch(greensFunction.getType()){
	case Property::GreensFunction::Type::Advanced:
	case Property::GreensFunction::Type::Retarded:
	{
		const Range &energyWindow
			= propertyExtractor->getEnergyWindow();
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

		for(unsigned int e = 0; e < energyWindow.getResolution(); e++){
			double E = energyWindow[e];;

			for(int n = 0; n < model.getBasisSize(); n++){
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
		double chemicalPotential = model.getChemicalPotential();

		for(unsigned int e = 0; e < numMatsubaraEnergies; e++){
			complex<double> E = greensFunction.getMatsubaraEnergy(e)
				+ chemicalPotential;

			for(int n = 0; n < model.getBasisSize(); n++){
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
	const Solver::Diagonalizer &solver = propertyExtractor->getSolver();
	const Model &model = solver.getModel();

	const CArray<double> &eigenValues = solver.getEigenValues();
	for(int n = 0; n < model.getBasisSize(); n++){
		double weight = getThermodynamicEquilibriumOccupation(
			eigenValues[n],
			model
		);
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
	const Solver::Diagonalizer &solver = propertyExtractor->getSolver();
	const Model &model = solver.getModel();

	const CArray<double> &eigenValues = solver.getEigenValues();

	int spinIndex = information.getSpinIndex();
	Index index_u(index);
	Index index_d(index);
	index_u.at(spinIndex) = 0;
	index_d.at(spinIndex) = 1;
	for(int n = 0; n < model.getBasisSize(); n++){
		double weight = getThermodynamicEquilibriumOccupation(
			eigenValues[n],
			model
		);

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
	const Solver::Diagonalizer &solver = propertyExtractor->getSolver();

	const CArray<double> &eigenValues = solver.getEigenValues();

	const Range &energyWindow = propertyExtractor->getEnergyWindow();
	double dE = ldos.getDeltaE();
	for(unsigned int n = 0; n < eigenValues.getSize(); n++){
		if(eigenValues[n] > energyWindow[0] && eigenValues[n] < energyWindow.getLast()){
			complex<double> u = solver.getAmplitude(n, index);

			int e = round((eigenValues[n] - energyWindow[0])/dE);
			if(e >= 0 && e < (int)energyWindow.getResolution())
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
	const Solver::Diagonalizer &solver = propertyExtractor->getSolver();
	const Model &model = solver.getModel();

	const CArray<double> &eigenValues = solver.getEigenValues();

	int spinIndex = information.getSpinIndex();

	Index index_u(index);
	Index index_d(index);
	index_u.at(spinIndex) = 0;
	index_d.at(spinIndex) = 1;
	const Range &energyWindow = propertyExtractor->getEnergyWindow();
	double dE = spinPolarizedLDOS.getDeltaE();
	for(int n = 0; n < model.getBasisSize(); n++){
		if(
			eigenValues[n] > energyWindow[0]
			&& eigenValues[n] < energyWindow.getLast()
		){
			complex<double> u_u = solver.getAmplitude(n, index_u);
			complex<double> u_d = solver.getAmplitude(n, index_d);

			unsigned int e = (unsigned int)(
				(eigenValues[n] - energyWindow[0])/dE
			);
			if(e >= energyWindow.getResolution())
				e = energyWindow.getResolution() - 1;
			data[offset + e].at(0, 0) += conj(u_u)*u_u/dE;
			data[offset + e].at(0, 1) += conj(u_u)*u_d/dE;
			data[offset + e].at(1, 0) += conj(u_d)*u_u/dE;
			data[offset + e].at(1, 1) += conj(u_d)*u_d/dE;
		}
	}
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
