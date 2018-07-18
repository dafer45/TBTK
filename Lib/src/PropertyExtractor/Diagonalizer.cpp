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
#include "TBTK/Functions.h"
#include "TBTK/Streams.h"

#include <cmath>

using namespace std;

static complex<double> i(0, 1);

namespace TBTK{
namespace PropertyExtractor{

Diagonalizer::Diagonalizer(Solver::Diagonalizer &dSolver){
	this->dSolver = &dSolver;
}

/*Diagonalizer::~Diagonalizer(){
}*/

/*void Diagonalizer::saveEigenValues(string path, string filename){
	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << '/';
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str());
	for(int n = 0; n < dSolver->getModel().getBasisSize(); n++){
		fout << dSolver->getEigenValues()[n] << "\n";
	}
	fout.close();
}

void Diagonalizer::getTabulatedHoppingAmplitudeSet(
	complex<double> **amplitudes,
	int **indices,
	int *numHoppingAmplitudes,
	int *maxIndexSize
){
	dSolver->getModel().getHoppingAmplitudeSet().tabulate(
		amplitudes,
		indices,
		numHoppingAmplitudes,
		maxIndexSize
	);
}*/

Property::EigenValues Diagonalizer::getEigenValues(){
	int size = dSolver->getModel().getBasisSize();
	const double *ev = dSolver->getEigenValues();

	Property::EigenValues eigenValues(size);
	std::vector<double> &data = eigenValues.getDataRW();
	for(int n = 0; n < size; n++)
		data[n] = ev[n];

	return eigenValues;
}

Property::WaveFunctions Diagonalizer::calculateWaveFunctions(
	initializer_list<Index> patterns,
	vector<int> states
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

	vector<unsigned int> statesVector;
	if(states.size() == 1){
		if(*states.begin() == IDX_ALL){
			for(int n = 0; n < dSolver->getModel().getBasisSize(); n++)
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

	hint = new Property::WaveFunctions*[1];
	((Property::WaveFunctions**)hint)[0] = &waveFunctions;

	calculate(
		calculateWaveFunctionsCallback,
		allIndices,
		memoryLayout,
		waveFunctions
	);

	delete [] (Property::WaveFunctions**)hint;

	return waveFunctions;
}

/*Property::GreensFunction* DPropertyExtractor::calculateGreensFunction(
	Index to,
	Index from,
	Property::GreensFunction::Type type
){
	unsigned int numPoles = dSolver->getModel().getBasisSize();

	complex<double> *positions = new complex<double>[numPoles];
	complex<double> *amplitudes = new complex<double>[numPoles];
	for(int n = 0; n < dSolver->getModel().getBasisSize(); n++){
		positions[n] = dSolver->getEigenValue(n);

		complex<double> uTo = dSolver->getAmplitude(n, to);
		complex<double> uFrom = dSolver->getAmplitude(n, from);

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

Property::GreensFunction Diagonalizer::calculateGreensFunction(
	std::initializer_list<Index> patterns,
	Property::GreensFunction::Type type
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
		true,
		true
	);

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

		hint = &greensFunction;

		calculate(
			calculateGreensFunctionCallback,
			allIndices,
			memoryLayout,
			greensFunction
		);

		hint = nullptr;

		break;
	}
	default:
		TBTKExit(
			"PropertyExtractor::Diagonalizer::calculateGreensFunction()",
			"Only type Property::GreensFunction::Type::Advanced"
			<< " and Property::GrensFunction::Type::Retarded"
			<< " supported yet.",
			""
		);
	}

	return greensFunction;
}

Property::DOS Diagonalizer::calculateDOS(){
	const double *ev = dSolver->getEigenValues();

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	Property::DOS dos(lowerBound, upperBound, energyResolution);
	std::vector<double> &data = dos.getDataRW();
	double dE = (upperBound - lowerBound)/energyResolution;
	for(int n = 0; n < dSolver->getModel().getBasisSize(); n++){
		int e = (int)(((ev[n] - lowerBound)/(upperBound - lowerBound))*energyResolution);
		if(e >= 0 && e < energyResolution){
			data[e] += 1./dE;
		}
	}

	return dos;
}

complex<double> Diagonalizer::calculateExpectationValue(
	Index to,
	Index from
){
	const complex<double> i(0, 1);

	complex<double> expectationValue = 0.;

	Statistics statistics = dSolver->getModel().getStatistics();

	for(int n = 0; n < dSolver->getModel().getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				dSolver->getEigenValue(n),
				dSolver->getModel().getChemicalPotential(),
				dSolver->getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				dSolver->getEigenValue(n),
				dSolver->getModel().getChemicalPotential(),
				dSolver->getModel().getTemperature()
			);
		}

		complex<double> u_to = dSolver->getAmplitude(n, to);
		complex<double> u_from = dSolver->getAmplitude(n, from);

		expectationValue += weight*conj(u_to)*u_from;
	}

	return expectationValue;
}

Property::Density Diagonalizer::calculateDensity(
	Index pattern,
	Index ranges
){
	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Density density(lDimensions, lRanges);

	calculate(calculateDensityCallback, (void*)density.getDataRW().data(), pattern, ranges, 0, 1);

	return density;
}

Property::Density Diagonalizer::calculateDensity(
	initializer_list<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
		false,
		false
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
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

Property::Magnetization Diagonalizer::calculateMagnetization(
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
			"PropertyExtractor::Diagonalizer::calculateMagnetization()",
			"No spin index indiceated.",
			"Used IDX_SPIN to indicate the position of the spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Magnetization magnetization(lDimensions, lRanges);

	calculate(
		calculateMAGCallback,
		(void*)magnetization.getDataRW().data(),
		pattern,
		ranges,
		0,
		1
	);

	delete [] (int*)hint;

	return magnetization;
}

Property::Magnetization Diagonalizer::calculateMagnetization(
	std::initializer_list<Index> patterns
){
	IndexTree allIndices = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
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

Property::LDOS Diagonalizer::calculateLDOS(
	Index pattern,
	Index ranges
){
	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	//hint[0] is an array of doubles, hint[1] is an array of ints
	//hint[0][0]: upperBound
	//hint[0][1]: lowerBound
	//hint[1][0]: resolution
	//hint[1][1]: spin_index
	hint = new void*[2];
	((double**)hint)[0] = new double[2];
	((int**)hint)[1] = new int[1];
	((double**)hint)[0][0] = upperBound;
	((double**)hint)[0][1] = lowerBound;
	((int**)hint)[1][0] = energyResolution;

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

	calculate(calculateLDOSCallback, (void*)ldos.getDataRW().data(), pattern, ranges, 0, energyResolution);

	return ldos;
}

Property::LDOS Diagonalizer::calculateLDOS(
	std::initializer_list<Index> patterns
){
	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	//hint[0] is an array of doubles, hint[1] is an array of ints
	//hint[0][0]: upperBound
	//hint[0][1]: lowerBound
	//hint[1][0]: resolution
	//hint[1][1]: spin_index
	hint = new void*[2];
	((double**)hint)[0] = new double[2];
	((int**)hint)[1] = new int[1];
	((double**)hint)[0][0] = upperBound;
	((double**)hint)[0][1] = lowerBound;
	((int**)hint)[1][0] = energyResolution;

	IndexTree allIndices = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
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

Property::SpinPolarizedLDOS Diagonalizer::calculateSpinPolarizedLDOS(
	Index pattern,
	Index ranges
){
	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	//hint[0] is an array of doubles, hint[1] is an array of ints
	//hint[0][0]: upperBound
	//hint[0][1]: lowerBound
	//hint[1][0]: resolution
	//hint[1][1]: spin_index
	hint = new void*[2];
	((double**)hint)[0] = new double[2];
	((int**)hint)[1] = new int[2];
	((double**)hint)[0][0] = upperBound;
	((double**)hint)[0][1] = lowerBound;
	((int**)hint)[1][0] = energyResolution;

	((int**)hint)[1][1] = -1;
	for(unsigned int n = 0; n < pattern.getSize(); n++){
		if(pattern.at(n) == IDX_SPIN){
			((int**)hint)[1][1] = n;
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(((int**)hint)[1][1] == -1){
		delete [] ((double**)hint)[0];
		delete [] ((int**)hint)[1];
		delete [] (void**)hint;
		TBTKExit(
			"PropertyExtractor::Diagonalizer::calculateSpinPolarizedLDOS()",
			"No spin index indicated.",
			"Used IDX_SPIN to indicate the position of the spin index."
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
		(void*)spinPolarizedLDOS.getDataRW().data(),
		pattern,
		ranges,
		0,
		energyResolution
	);

	delete [] ((double**)hint)[0];
	delete [] ((int**)hint)[1];
	delete [] (void**)hint;

	return spinPolarizedLDOS;
}

Property::SpinPolarizedLDOS Diagonalizer::calculateSpinPolarizedLDOS(
	std::initializer_list<Index> patterns
){
	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	//hint[0] is an array of doubles, hint[1] is an array of ints
	//hint[0][0]: upperBound
	//hint[0][1]: lowerBound
	//hint[1][0]: resolution
	//hint[1][1]: spin_index
	hint = new void*[2];
	((double**)hint)[0] = new double[2];
	((int**)hint)[1] = new int[2];
	((double**)hint)[0][0] = upperBound;
	((double**)hint)[0][1] = lowerBound;
	((int**)hint)[1][0] = energyResolution;

	IndexTree allIndices = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
		false,
		true
	);

	IndexTree memoryLayout = generateIndexTree(
		patterns,
		dSolver->getModel().getHoppingAmplitudeSet(),
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
		&(((int**)hint)[1][1])
	);

	delete [] ((double**)hint)[0];
	delete [] ((int**)hint)[1];
	delete [] (void**)hint;

	return spinPolarizedLDOS;
}

double Diagonalizer::calculateEntropy(){
	Statistics statistics = dSolver->getModel().getStatistics();

	double entropy = 0.;
	for(int n = 0; n < dSolver->getModel().getBasisSize(); n++){
		double p;

		switch(statistics){
		case Statistics::FermiDirac:
			p = Functions::fermiDiracDistribution(
				getEigenValue(n),
				dSolver->getModel().getChemicalPotential(),
				dSolver->getModel().getTemperature()
			);
			break;
		case Statistics::BoseEinstein:
			p = Functions::boseEinsteinDistribution(
				getEigenValue(n),
				dSolver->getModel().getChemicalPotential(),
				dSolver->getModel().getTemperature()
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

	entropy *= UnitHandler::getK_BN();

	return entropy;
}

void Diagonalizer::calculateWaveFunctionsCallback(
	PropertyExtractor *cb_this,
	void *waveFunctions,
	const Index &index,
	int offset
){
	Diagonalizer *pe = (Diagonalizer*)cb_this;

	const vector<unsigned int> states = ((Property::WaveFunctions**)pe->hint)[0]->getStates();
	for(unsigned int n = 0; n < states.size(); n++)
		((complex<double>*)waveFunctions)[offset + n] += pe->getAmplitude(states.at(n), index);
}

void Diagonalizer::calculateGreensFunctionCallback(
	PropertyExtractor *cb_this,
	void *greensFunction,
	const Index &index,
	int offset
){
	Diagonalizer *pe = (Diagonalizer*)cb_this;

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();
	double dE = (upperBound - lowerBound)/energyResolution;
	double delta = pe->getEnergyInfinitesimal();

	vector<Index> components = index.split();

	Property::GreensFunction &gf = *(Property::GreensFunction*)pe->hint;

	switch(gf.getType()){
	case Property::GreensFunction::Type::Advanced:
		delta *= -1;
	case Property::GreensFunction::Type::Retarded:
	{
		for(int e = 0; e < energyResolution; e++){
			double E = lowerBound + e*dE;
			for(
				int n = 0;
				n < pe->dSolver->getModel().getBasisSize();
				n++
			){
				double E_n = pe->getEigenValue(n);
				complex<double> amplitude0
					= pe->getAmplitude(n, components[0]);
				complex<double> amplitude1
					= pe->getAmplitude(n, components[1]);
				((complex<double>*)greensFunction)[offset + e]
					+= amplitude0*conj(amplitude1)/(
						E - E_n + i*delta
					);
			}
		}

		break;
	}
	default:
		TBTKExit(
			"Diagonalizer::calculateGreensFunctionCallback()",
			"Only type GreensFunction::Type::Advanced and"
			<< " GreensFunction::Type::Retarded supported yet.",
			""
		);
	}
}

void Diagonalizer::calculateDensityCallback(
	PropertyExtractor *cb_this,
	void* density,
	const Index &index,
	int offset
){
	Diagonalizer *pe = (Diagonalizer*)cb_this;

	const double *eigen_values = pe->dSolver->getEigenValues();
	Statistics statistics = pe->dSolver->getModel().getStatistics();
	for(int n = 0; n < pe->dSolver->getModel().getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(
				eigen_values[n],
				pe->dSolver->getModel().getChemicalPotential(),
				pe->dSolver->getModel().getTemperature()
			);
		}
		else{
			weight = Functions::boseEinsteinDistribution(
				eigen_values[n],
				pe->dSolver->getModel().getChemicalPotential(),
				pe->dSolver->getModel().getTemperature()
			);
		}

		complex<double> u = pe->dSolver->getAmplitude(n, index);

		((double*)density)[offset] += pow(abs(u), 2)*weight;
	}
}

void Diagonalizer::calculateMAGCallback(
	PropertyExtractor *cb_this,
	void *mag,
	const Index &index,
	int offset
){
	Diagonalizer *pe = (Diagonalizer*)cb_this;

	const double *eigen_values = pe->dSolver->getEigenValues();
	Statistics statistics = pe->dSolver->getModel().getStatistics();

	int spin_index = ((int*)pe->hint)[0];
	Index index_u(index);
	Index index_d(index);
	index_u.at(spin_index) = 0;
	index_d.at(spin_index) = 1;
	for(int n = 0; n < pe->dSolver->getModel().getBasisSize(); n++){
		double weight;
		if(statistics == Statistics::FermiDirac){
			weight = Functions::fermiDiracDistribution(eigen_values[n],
									pe->dSolver->getModel().getChemicalPotential(),
									pe->dSolver->getModel().getTemperature());
		}
		else{
			weight = Functions::boseEinsteinDistribution(eigen_values[n],
									pe->dSolver->getModel().getChemicalPotential(),
									pe->dSolver->getModel().getTemperature());
		}

		complex<double> u_u = pe->dSolver->getAmplitude(n, index_u);
		complex<double> u_d = pe->dSolver->getAmplitude(n, index_d);

		((SpinMatrix*)mag)[offset].at(0, 0) += conj(u_u)*u_u*weight;
		((SpinMatrix*)mag)[offset].at(0, 1) += conj(u_u)*u_d*weight;
		((SpinMatrix*)mag)[offset].at(1, 0) += conj(u_d)*u_u*weight;
		((SpinMatrix*)mag)[offset].at(1, 1) += conj(u_d)*u_d*weight;
	}
}

void Diagonalizer::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	void *ldos,
	const Index &index,
	int offset
){
	Diagonalizer *pe = (Diagonalizer*)cb_this;

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();

	const double *eigen_values = pe->dSolver->getEigenValues();

	double u_lim = ((double**)pe->hint)[0][0];
	double l_lim = ((double**)pe->hint)[0][1];
	int resolution = ((int**)pe->hint)[1][0];

	double step_size = (u_lim - l_lim)/(double)resolution;

	double dE = (upperBound - lowerBound)/energyResolution;
	for(int n = 0; n < pe->dSolver->getModel().getBasisSize(); n++){
		if(eigen_values[n] > l_lim && eigen_values[n] < u_lim){
			complex<double> u = pe->dSolver->getAmplitude(n, index);

			int e = (int)((eigen_values[n] - l_lim)/step_size);
			if(e >= resolution)
				e = resolution-1;
			((double*)ldos)[offset + e] += real(conj(u)*u)/dE;
		}
	}
}

void Diagonalizer::calculateSP_LDOSCallback(
	PropertyExtractor *cb_this,
	void *sp_ldos,
	const Index &index,
	int offset
){
	Diagonalizer *pe = (Diagonalizer*)cb_this;

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();

	const double *eigen_values = pe->dSolver->getEigenValues();

	double u_lim = ((double**)pe->hint)[0][0];
	double l_lim = ((double**)pe->hint)[0][1];
	int resolution = ((int**)pe->hint)[1][0];
	int spin_index = ((int**)pe->hint)[1][1];

	double step_size = (u_lim - l_lim)/(double)resolution;

	Index index_u(index);
	Index index_d(index);
	index_u.at(spin_index) = 0;
	index_d.at(spin_index) = 1;
	double dE = (upperBound - lowerBound)/energyResolution;
	for(int n = 0; n < pe->dSolver->getModel().getBasisSize(); n++){
		if(eigen_values[n] > l_lim && eigen_values[n] < u_lim){
			complex<double> u_u = pe->dSolver->getAmplitude(n, index_u);
			complex<double> u_d = pe->dSolver->getAmplitude(n, index_d);

			int e = (int)((eigen_values[n] - l_lim)/step_size);
			if(e >= resolution)
				e = resolution-1;
			((SpinMatrix*)sp_ldos)[offset + e].at(0, 0) += conj(u_u)*u_u/dE;
			((SpinMatrix*)sp_ldos)[offset + e].at(0, 1) += conj(u_u)*u_d/dE;
			((SpinMatrix*)sp_ldos)[offset + e].at(1, 0) += conj(u_d)*u_u/dE;
			((SpinMatrix*)sp_ldos)[offset + e].at(1, 1) += conj(u_d)*u_d/dE;
		}
	}
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
