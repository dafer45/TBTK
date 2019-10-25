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
#include "TBTK/Functions.h"
#include "TBTK/Streams.h"

#include <cmath>

using namespace std;

static complex<double> i(0, 1);

namespace TBTK{
namespace PropertyExtractor{

BlockDiagonalizer::BlockDiagonalizer(Solver::BlockDiagonalizer &bSolver){
	this->bSolver = &bSolver;
}

BlockDiagonalizer::~BlockDiagonalizer(){
}

Property::EigenValues BlockDiagonalizer::getEigenValues(){
	int size = bSolver->getModel().getBasisSize();

	Property::EigenValues eigenValues(size);
	std::vector<double> &data = eigenValues.getDataRW();
	for(int n = 0; n < size; n++)
		data[n] = bSolver->getEigenValue(n);

	return eigenValues;
}

Property::WaveFunctions BlockDiagonalizer::calculateWaveFunctions(
	vector<Index> patterns,
	vector<Subindex> states
){
	validatePatternsNumComponents(
		patterns,
		1,
		"PropertyExtractor::BlockDiagonalizer::calculateWaveFunction()"
	);
	validatePatternsSpecifiers(
		patterns,
		{IDX_ALL, IDX_SUM_ALL},
		"PropertyExtractor::BlockDiagonalizer::calculateWaveFunction()"
	);

	IndexTree allIndices = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	vector<unsigned int> statesVector;
	if(states.size() == 1){
		if((*states.begin()).isWildcard()){
			for(int n = 0; n < bSolver->getModel().getBasisSize(); n++)
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
	validatePatternsNumComponents(
		patterns,
		2,
		"PropertyExtractor::BlockDiagonalizer::calculateGreensFunction()"
	);
	validatePatternsSpecifiers(
		patterns,
		{IDX_ALL, IDX_SUM_ALL},
		"PropertyExtractor::BlockDiagonalizer::calculateGreensFunction()"
	);

	IndexTree allIndices = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
		true,
		false
	);

	switch(type){
	case Property::GreensFunction::Type::Advanced:
	case Property::GreensFunction::Type::Retarded:
	{
		Property::GreensFunction greensFunction(
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

		double temperature = bSolver->getModel().getTemperature();
		double kT = UnitHandler::getK_BN()*temperature;
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

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	Property::DOS dos(lowerBound, upperBound, energyResolution);
	std::vector<double> &data = dos.getDataRW();
	double dE = dos.getDeltaE();
	for(int n = 0; n < bSolver->getModel().getBasisSize(); n++){
		int e = round((bSolver->getEigenValue(n) - lowerBound)/dE);
		if(e >= 0 && e < energyResolution)
			data[e] += 1./dE;
	}

	return dos;
}

complex<double> BlockDiagonalizer::calculateExpectationValue(
	Index to,
	Index from
){
	const complex<double> i(0, 1);

	complex<double> expectationValue = 0.;

	Statistics statistics = bSolver->getModel().getStatistics();

	for(int n = 0; n < bSolver->getModel().getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				bSolver->getEigenValue(n),
				bSolver->getModel().getChemicalPotential(),
				bSolver->getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				bSolver->getEigenValue(n),
				bSolver->getModel().getChemicalPotential(),
				bSolver->getModel().getTemperature()
			);
		}

		complex<double> u_to = bSolver->getAmplitude(n, to);
		complex<double> u_from = bSolver->getAmplitude(n, from);

		expectationValue += weight*conj(u_to)*u_from;
	}

	return expectationValue;
}

Property::Density BlockDiagonalizer::calculateDensity(
	vector<Index> patterns
){
	validatePatternsNumComponents(
		patterns,
		1,
		"PropertyExtractor::BlockDiagonalizer::calculateDensity()"
	);
	validatePatternsSpecifiers(
		patterns,
		{IDX_ALL, IDX_SUM_ALL},
		"PropertyExtractor::BlockDiagonalizer::calculateDensity()"
	);

	IndexTree allIndices = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
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

Property::Magnetization BlockDiagonalizer::calculateMagnetization(
	vector<Index> patterns
){
	validatePatternsNumComponents(
		patterns,
		1,
		"PropertyExtractor::BlockDiagonalizer::calculateMagnetization()"
	);
	validatePatternsSpecifiers(
		patterns,
		{IDX_ALL, IDX_SUM_ALL, IDX_SPIN},
		"PropertyExtractor::BlockDiagonalizer::calculateMagnetization()"
	);

	IndexTree allIndices = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

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
	validatePatternsNumComponents(
		patterns,
		1,
		"PropertyExtractor::BlockDiagonalizer::calculateLDOS()"
	);
	validatePatternsSpecifiers(
		patterns,
		{IDX_ALL, IDX_SUM_ALL},
		"PropertyExtractor::BlockDiagonalizer::calculateLDOS()"
	);

	TBTKAssert(
		getEnergyType() == EnergyType::Real,
		"PropertyExtractor::BlockDiagonalizer::calculateLDOS()",
		"Only real energies supported for the LDOS.",
		"Use PropertyExtractor::BlockDiagonalizer::setEnergyWindow()"
		<< " to set a real energy window."
	);

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	IndexTree allIndices = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

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

Property::SpinPolarizedLDOS BlockDiagonalizer::calculateSpinPolarizedLDOS(
	vector<Index> patterns
){
	validatePatternsNumComponents(
		patterns,
		1,
		"PropertyExtractor::BlockDiagonalizer::calculateSpinPolarizedLDOS()"
	);
	validatePatternsSpecifiers(
		patterns,
		{IDX_ALL, IDX_SUM_ALL, IDX_SPIN},
		"PropertyExtractor::BlockDiagonalizer::calculateSpinPolarizedLDOS()"
	);

	TBTKAssert(
		getEnergyType() == EnergyType::Real,
		"PropertyExtractor::BlockDiagonalizer::calculateSpinPolarizedLDOS()",
		"Only real energies supported for the SpinPolarizedLDOS.",
		"Use PropertyExtractor::BlockDiagonalizer::setEnergyWindow()"
		<< " to set a real energy window."
	);

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	IndexTree allIndices = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		bSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

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

double BlockDiagonalizer::calculateEntropy(){
	Statistics statistics = bSolver->getModel().getStatistics();

	double entropy = 0.;
	for(int n = 0; n < bSolver->getModel().getBasisSize(); n++){
		double p;

		switch(statistics){
		case Statistics::FermiDirac:
			p = Functions::fermiDiracDistribution(
				getEigenValue(n),
				bSolver->getModel().getChemicalPotential(),
				bSolver->getModel().getTemperature()
			);
			break;
		case Statistics::BoseEinstein:
			p = Functions::boseEinsteinDistribution(
				getEigenValue(n),
				bSolver->getModel().getChemicalPotential(),
				bSolver->getModel().getTemperature()
			);
			break;
		default:
			TBTKExit(
				"PropertyExtractor::BlockDiagonalizer::calculateEntropy()",
				"Unknow statistsics.",
				"This should never happen, contact the developer."
			);
		}

		entropy -= p*log(p);
	}

	entropy *= UnitHandler::getK_BN();

	return entropy;
}

void BlockDiagonalizer::calculateWaveFunctionsCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	BlockDiagonalizer *pe = (BlockDiagonalizer*)cb_this;
	Property::WaveFunctions &waveFunctions
		= (Property::WaveFunctions&)property;
	vector<complex<double>> &data = waveFunctions.getDataRW();

	const vector<unsigned int> states = waveFunctions.getStates();
	for(unsigned int n = 0; n < states.size(); n++)
		data[offset + n] += pe->getAmplitude(states.at(n), index);
}

void BlockDiagonalizer::calculateGreensFunctionCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	BlockDiagonalizer *propertyExtractor = (BlockDiagonalizer*)cb_this;
	Property::GreensFunction &greensFunction
		= (Property::GreensFunction&)property;
	vector<complex<double>> &data = greensFunction.getDataRW();

	vector<Index> components = index.split();
	const Index &toIndex = components[0];
	const Index &fromIndex = components[1];

	unsigned int firstStateInBlock
		= propertyExtractor->bSolver->getFirstStateInBlock(
			toIndex
		);
	unsigned int lastStateInBlock
		= propertyExtractor->bSolver->getLastStateInBlock(
			toIndex
		);
	if(
		firstStateInBlock
			!= propertyExtractor->bSolver->getFirstStateInBlock(
				fromIndex
			)
	){
		return;
	}

	switch(greensFunction.getType()){
	case Property::GreensFunction::Type::Advanced:
	case Property::GreensFunction::Type::Retarded:
	{
		double lowerBound = propertyExtractor->getLowerBound();
		double upperBound = propertyExtractor->getUpperBound();
		double energyResolution
			= propertyExtractor->getEnergyResolution();
		double dE = greensFunction.getDeltaE();
		double delta = propertyExtractor->getEnergyInfinitesimal();
		if(greensFunction.getType() == Property::GreensFunction::Type::Advanced)
			delta *= -1;

		for(int e = 0; e < energyResolution; e++){
			double E = lowerBound + e*dE;
			for(
				int n = 0;
				n < propertyExtractor->bSolver->getModel(
					).getBasisSize();
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
			= propertyExtractor->bSolver->getModel(
			).getChemicalPotential();

		for(
			unsigned int n = firstStateInBlock;
			n <= lastStateInBlock;
			n++
		){
			double energy
				= propertyExtractor->bSolver->getEigenValue(n)
					- chemicalPotential;

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
	BlockDiagonalizer *pe = (BlockDiagonalizer*)cb_this;
	Property::Density &density = (Property::Density&)property;
	vector<double> &data = density.getDataRW();

	Statistics statistics = pe->bSolver->getModel().getStatistics();
	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				pe->bSolver->getEigenValue(n),
				pe->bSolver->getModel().getChemicalPotential(),
				pe->bSolver->getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				pe->bSolver->getEigenValue(n),
				pe->bSolver->getModel().getChemicalPotential(),
				pe->bSolver->getModel().getTemperature()
			);
		}

		complex<double> u = pe->bSolver->getAmplitude(n, index);

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
	BlockDiagonalizer *pe = (BlockDiagonalizer*)cb_this;
	Property::Magnetization &magnetization
		= (Property::Magnetization&)property;
	vector<SpinMatrix> &data = magnetization.getDataRW();

	Statistics statistics = pe->bSolver->getModel().getStatistics();

	int spinIndex = information.getSpinIndex();
	Index index_u(index);
	Index index_d(index);
	index_u.at(spinIndex) = 0;
	index_d.at(spinIndex) = 1;
	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				pe->bSolver->getEigenValue(n),
				pe->bSolver->getModel().getChemicalPotential(),
				pe->bSolver->getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				pe->bSolver->getEigenValue(n),
				pe->bSolver->getModel().getChemicalPotential(),
				pe->bSolver->getModel().getTemperature()
			);
		}

		complex<double> u_u = pe->bSolver->getAmplitude(n, index_u);
		complex<double> u_d = pe->bSolver->getAmplitude(n, index_d);

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
	BlockDiagonalizer *pe = (BlockDiagonalizer*)cb_this;
	Property::LDOS &ldos = (Property::LDOS&)property;
	vector<double> &data = ldos.getDataRW();

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();

	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	double dE = ldos.getDeltaE();
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
		double eigenValue = pe->bSolver->getEigenValue(n);
		if(eigenValue > lowerBound && eigenValue < upperBound){
			complex<double> u = pe->bSolver->getAmplitude(n, index);

			int e = (int)((eigenValue - lowerBound)/dE);
			if(e >= energyResolution)
				e = energyResolution - 1;
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
	BlockDiagonalizer *pe = (BlockDiagonalizer*)cb_this;
	Property::SpinPolarizedLDOS &spinPolarizedLDOS
		= (Property::SpinPolarizedLDOS&)property;
	vector<SpinMatrix> &data = spinPolarizedLDOS.getDataRW();

	int spinIndex = information.getSpinIndex();

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();

	Index index_u(index);
	Index index_d(index);
	index_u.at(spinIndex) = 0;
	index_d.at(spinIndex) = 1;
	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	double dE = spinPolarizedLDOS.getDeltaE();
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
		double eigenValue = pe->bSolver->getEigenValue(n);
		if(eigenValue > lowerBound && eigenValue < upperBound){
			complex<double> u_u = pe->bSolver->getAmplitude(n, index_u);
			complex<double> u_d = pe->bSolver->getAmplitude(n, index_d);

			int e = (int)((eigenValue - lowerBound)/dE);
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
