/* Copyright 2017 Kristofer Björnson
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

/** @file BlockDiagonalizer.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PropertyExtractor/BlockDiagonalizer.h"
#include "TBTK/PropertyExtractor/IndexTreeGenerator.h"
#include "TBTK/PropertyExtractor/PatternValidator.h"
#include "TBTK/Functions.h"
#include "TBTK/Streams.h"

#include <cmath>

using namespace std;

static complex<double> i(0, 1);

namespace TBTK{
namespace PropertyExtractor{

BlockDiagonalizer::BlockDiagonalizer(){
}

Property::EigenValues BlockDiagonalizer::getEigenValues(){
	const Solver::BlockDiagonalizer &solver = getSolver();
	int size = solver.getModel().getBasisSize();

	Property::EigenValues eigenValues(size);
	std::vector<double> &data = eigenValues.getDataRW();
	for(int n = 0; n < size; n++)
		data[n] = solver.getEigenValue(n);

	return eigenValues;
}

Property::WaveFunctions BlockDiagonalizer::calculateWaveFunctions(
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
				"PropertyExtractor::BlockDiagonalizer::calculateWaveFunctions()",
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
				"PropertyExtractor::BlockDiagonalizer::calculateWaveFunctions()",
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

/*Property::GreensFunction2 BPropertyExtractor::calculateGreensFunction(
	Index to,
	Index from,
	Property::GreensFunction2::Type type
){
	unsigned int numPoles = bSolver->getModel().getBasisSize();

	complex<double> *positions = new complex<double>[numPoles];
	complex<double> *amplitudes = new complex<double>[numPoles];
	for(int n = 0; n < bSolver->getModel().getBasisSize(); n++){
		positions[n] = bSolver->getEigenValue(n);

		complex<double> uTo = bSolver->getAmplitude(n, to);
		complex<double> uFrom = bSolver->getAmplitude(n, from);

		amplitudes[n] = uTo*conj(uFrom);
	}

	Property::GreensFunction *greensFunction = new Property::GreensFunction(
		type,
		Property::GreensFunction::Format::Poles,
		numPoles,
		positions,
		amplitudes
	);

	delete [] positions;
	delete [] amplitudes;

	return greensFunction;
}*/

Property::GreensFunction BlockDiagonalizer::calculateGreensFunction(
	vector<Index> patterns,
	Property::GreensFunction::Type type
){
	PatternValidator::validateGreensFunctionPatterns(patterns);
	IndexTree allIndices = generateAllIndices(patterns);
	IndexTree memoryLayout = generateMemoryLayout(patterns);
	const Model &model = getSolver().getModel();

	switch(type){
	case Property::GreensFunction::Type::Advanced:
	case Property::GreensFunction::Type::Retarded:
	{
		Property::GreensFunction greensFunction(
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

		return greensFunction;
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
			"PropertyExtractor::BlockDiagonalizer::calculateGreensFunction()",
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

		Property::GreensFunction greensFunction(
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

		return greensFunction;
	}
	default:
		TBTKExit(
			"PropertyExtractor::BlockDiagonalizer::calculateGreensFunction()",
			"Only Property::GreensFunction::Type Advanced,"
			<< " Retarded, and Matsubara supported yet.",
			""
		);
	}
}

Property::DOS BlockDiagonalizer::calculateDOS(){
	TBTKAssert(
		getEnergyType() == EnergyType::Real,
		"PropertyExtractor::BlockDiagonalizer::calculateDOS()",
		"Only real energies supported for the DOS.",
		"Use PropertyExtractor::BlockDiagonalizer::setEnergyWindow()"
		<< " to set a real energy window."
	);

	const Range &energyWindow = getEnergyWindow();
	Property::DOS dos(energyWindow);
	std::vector<double> &data = dos.getDataRW();
	double dE = dos.getDeltaE();
	const Solver::BlockDiagonalizer &solver = getSolver();
	for(int n = 0; n < solver.getModel().getBasisSize(); n++){
		int e = round((solver.getEigenValue(n) - energyWindow[0])/dE);
		if(e >= 0 && e < (int)energyWindow.getResolution())
			data[e] += 1./dE;
	}

	return dos;
}

complex<double> BlockDiagonalizer::calculateExpectationValue(
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

Property::Density BlockDiagonalizer::calculateDensity(
	vector<Index> patterns
){
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

Property::Magnetization BlockDiagonalizer::calculateMagnetization(
	vector<Index> patterns
){
	PatternValidator::validateMagnetizationPatterns(patterns);
	IndexTree allIndices = generateAllIndices(patterns);
	IndexTree memoryLayout = generateMemoryLayout(patterns);

	Property::Magnetization magnetization(memoryLayout);
	Information information;
	calculate(
		calculateMagnetizationCallback,
		allIndices,
		memoryLayout,
		magnetization,
		information
	);

	return magnetization;
}

Property::LDOS BlockDiagonalizer::calculateLDOS(
	vector<Index> patterns
){
	PatternValidator::validateLDOSPatterns(patterns);
	TBTKAssert(
		getEnergyType() == EnergyType::Real,
		"PropertyExtractor::BlockDiagonalizer::calculateLDOS()",
		"Only real energies supported for the LDOS.",
		"Use PropertyExtractor::BlockDiagonalizer::setEnergyWindow()"
		<< " to set a real energy window."
	);
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

Property::SpinPolarizedLDOS BlockDiagonalizer::calculateSpinPolarizedLDOS(
	vector<Index> patterns
){
	PatternValidator::validateSpinPolarizedLDOSPatterns(patterns);
	TBTKAssert(
		getEnergyType() == EnergyType::Real,
		"PropertyExtractor::BlockDiagonalizer::calculateSpinPolarizedLDOS()",
		"Only real energies supported for the SpinPolarizedLDOS.",
		"Use PropertyExtractor::BlockDiagonalizer::setEnergyWindow()"
		<< " to set a real energy window."
	);
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

double BlockDiagonalizer::calculateEntropy(){
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

void BlockDiagonalizer::calculateWaveFunctionsCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	BlockDiagonalizer *propertyExtractor = (BlockDiagonalizer*)cb_this;
	Property::WaveFunctions &waveFunctions
		= (Property::WaveFunctions&)property;
	vector<complex<double>> &data = waveFunctions.getDataRW();

	const vector<unsigned int> states = waveFunctions.getStates();
	for(unsigned int n = 0; n < states.size(); n++)
		data[offset + n] += propertyExtractor->getAmplitude(states.at(n), index);
}

void BlockDiagonalizer::calculateGreensFunctionCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	BlockDiagonalizer *propertyExtractor = (BlockDiagonalizer*)cb_this;
	const Solver::BlockDiagonalizer &solver = propertyExtractor->getSolver();
	Property::GreensFunction &greensFunction
		= (Property::GreensFunction&)property;
	vector<complex<double>> &data = greensFunction.getDataRW();

	vector<Index> components = index.split();
	const Index &toIndex = components[0];
	const Index &fromIndex = components[1];

	unsigned int firstStateInBlock = solver.getFirstStateInBlock(toIndex);
	unsigned int lastStateInBlock = solver.getLastStateInBlock(toIndex);
	if(firstStateInBlock != solver.getFirstStateInBlock(fromIndex))
		return;

	switch(greensFunction.getType()){
	case Property::GreensFunction::Type::Advanced:
	case Property::GreensFunction::Type::Retarded:
	{
		const Range &energyWindow
			= propertyExtractor->getEnergyWindow();
		double delta = propertyExtractor->getEnergyInfinitesimal();
		if(greensFunction.getType() == Property::GreensFunction::Type::Advanced)
			delta *= -1;

		for(unsigned int e = 0; e < energyWindow.getResolution(); e++){
			double E = energyWindow[e];
			for(
				int n = 0;
				n < solver.getModel().getBasisSize();
				n++
			){
				double E_n
					= propertyExtractor->getEigenValue(n);
				complex<double> amplitude0
					= propertyExtractor->getAmplitude(
						n,
						components[0]
					);
				complex<double> amplitude1
					= propertyExtractor->getAmplitude(
						n,
						components[1]
					);
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
		double chemicalPotential
			= solver.getModel().getChemicalPotential();

		for(
			unsigned int n = firstStateInBlock;
			n <= lastStateInBlock;
			n++
		){
			double energy
				= solver.getEigenValue(n) - chemicalPotential;

			complex<double> toAmplitude = propertyExtractor->getAmplitude(
				n,
				toIndex
			);
			complex<double> fromAmplitude
				= propertyExtractor->getAmplitude(
					n,
					fromIndex
				);
			complex<double> numerator = toAmplitude*conj(fromAmplitude);
			if(abs(numerator) < numeric_limits<double>::epsilon())
				continue;

			for(unsigned int e = 0; e < numMatsubaraEnergies; e++){
				data[offset + e] += numerator/(
					greensFunction.getMatsubaraEnergy(e)
					- energy
				);
			}
		}

		break;
	}
	default:
		TBTKExit(
			"PropertyExtractor::BlockDiagonalizer::calculateGreensFunctionCallback()",
			"Unsupported Green's function type.",
			"This should never happen, contact the developer."
			//Should have been cought in calculateGreensFunction().
		);
	}
}

void BlockDiagonalizer::calculateDensityCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	BlockDiagonalizer *propertyExtractor = (BlockDiagonalizer*)cb_this;
	const Solver::BlockDiagonalizer &solver = propertyExtractor->getSolver();
	Property::Density &density = (Property::Density&)property;
	vector<double> &data = density.getDataRW();

	int firstStateInBlock = solver.getFirstStateInBlock(index);
	int lastStateInBlock = solver.getLastStateInBlock(index);
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
		double weight = getThermodynamicEquilibriumOccupation(
			solver.getEigenValue(n),
			solver.getModel()
		);
		complex<double> u = solver.getAmplitude(n, index);

		data[offset] += pow(abs(u), 2)*weight;
	}
}

void BlockDiagonalizer::calculateMagnetizationCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	BlockDiagonalizer *propertyExtractor = (BlockDiagonalizer*)cb_this;
	Solver::BlockDiagonalizer &solver = propertyExtractor->getSolver();
	Property::Magnetization &magnetization
		= (Property::Magnetization&)property;
	vector<SpinMatrix> &data = magnetization.getDataRW();

	int spinIndex = information.getSpinIndex();
	Index index_u(index);
	Index index_d(index);
	index_u.at(spinIndex) = 0;
	index_d.at(spinIndex) = 1;
	int firstStateInBlock = solver.getFirstStateInBlock(index);
	int lastStateInBlock = solver.getLastStateInBlock(index);
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
		double weight = getThermodynamicEquilibriumOccupation(
			solver.getEigenValue(n),
			solver.getModel()
		);

		complex<double> u_u = solver.getAmplitude(n, index_u);
		complex<double> u_d = solver.getAmplitude(n, index_d);

		data[offset].at(0, 0) += conj(u_u)*u_u*weight;
		data[offset].at(0, 1) += conj(u_u)*u_d*weight;
		data[offset].at(1, 0) += conj(u_d)*u_u*weight;
		data[offset].at(1, 1) += conj(u_d)*u_d*weight;
	}
}

void BlockDiagonalizer::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	BlockDiagonalizer *propertyExtractor = (BlockDiagonalizer*)cb_this;
	Solver::BlockDiagonalizer &solver = propertyExtractor->getSolver();
	Property::LDOS &ldos = (Property::LDOS&)property;
	vector<double> &data = ldos.getDataRW();

	int firstStateInBlock = solver.getFirstStateInBlock(index);
	int lastStateInBlock = solver.getLastStateInBlock(index);
	double dE = ldos.getDeltaE();
	const Range &energyWindow = propertyExtractor->getEnergyWindow();
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
		double eigenValue = solver.getEigenValue(n);
		if(
			eigenValue > energyWindow[0]
			&& eigenValue < energyWindow.getLast()
		){
			complex<double> u = solver.getAmplitude(n, index);

			int e = (int)((eigenValue - energyWindow[0])/dE);
			if(e >= (int)energyWindow.getResolution())
				e = energyWindow.getResolution() - 1;
			data[offset + e] += real(conj(u)*u)/dE;
		}
	}
}

void BlockDiagonalizer::calculateSP_LDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	BlockDiagonalizer *propertyExtractor = (BlockDiagonalizer*)cb_this;
	const Solver::BlockDiagonalizer &solver = propertyExtractor->getSolver();
	Property::SpinPolarizedLDOS &spinPolarizedLDOS
		= (Property::SpinPolarizedLDOS&)property;
	vector<SpinMatrix> &data = spinPolarizedLDOS.getDataRW();

	int spinIndex = information.getSpinIndex();

	Index index_u(index);
	Index index_d(index);
	index_u.at(spinIndex) = 0;
	index_d.at(spinIndex) = 1;
	int firstStateInBlock = solver.getFirstStateInBlock(index);
	int lastStateInBlock = solver.getLastStateInBlock(index);
	double dE = spinPolarizedLDOS.getDeltaE();
	const Range &energyWindow = propertyExtractor->getEnergyWindow();
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
		double eigenValue = solver.getEigenValue(n);
		if(
			eigenValue > energyWindow[0]
			&& eigenValue < energyWindow.getLast()
		){
			complex<double> u_u = solver.getAmplitude(n, index_u);
			complex<double> u_d = solver.getAmplitude(n, index_d);

			int e = (int)((eigenValue - energyWindow[0])/dE);
			if(e >= (int)energyWindow.getResolution())
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
