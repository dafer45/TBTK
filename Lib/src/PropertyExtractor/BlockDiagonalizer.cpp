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
	energyType = EnergyType::Real;
}

BlockDiagonalizer::~BlockDiagonalizer(){
}

void BlockDiagonalizer::setEnergyWindow(
	double lowerBound,
	double upperBound,
	int resolution
){
	PropertyExtractor::setEnergyWindow(
		lowerBound,
		upperBound,
		resolution
	);

	energyType = EnergyType::Real;
}

void BlockDiagonalizer::setEnergyWindow(
	int lowerFermionicMatsubaraEnergyIndex,
	int upperFermionicMatsubaraEnergyIndex,
	int lowerBosonicMatsubaraEnergyIndex,
	int upperBosonicMatsubaraEnergyIndex
){
	TBTKAssert(
		abs(lowerFermionicMatsubaraEnergyIndex%2) == 1,
		"PropertyExtractor::BlockDiagonalizer::setEnergyWindow()",
		"'lowerFermionicMatsubaraEnergyIndex="
		<< lowerFermionicMatsubaraEnergyIndex << "' must be odd.",
		""
	);
	TBTKAssert(
		abs(upperFermionicMatsubaraEnergyIndex%2) == 1,
		"PropertyExtractor::BlockDiagonalizer::setEnergyWindow()",
		"'upperFermionicMatsubaraEnergyIndex="
		<< upperFermionicMatsubaraEnergyIndex << "' must be odd.",
		""
	);
	TBTKAssert(
		abs(lowerBosonicMatsubaraEnergyIndex%2) == 0,
		"PropertyExtractor::BlockDiagonalizer::setEnergyWindow()",
		"'lowerBosonicMatsubaraEnergyIndex="
		<< lowerFermionicMatsubaraEnergyIndex << "' must be odd.",
		""
	);
	TBTKAssert(
		abs(upperBosonicMatsubaraEnergyIndex%2) == 0,
		"PropertyExtractor::BlockDiagonalizer::setEnergyWindow()",
		"'upperBosonicMatsubaraEnergyIndex="
		<< upperBosonicMatsubaraEnergyIndex << "' must be odd.",
		""
	);

	energyType = EnergyType::Matsubara;
	this->lowerFermionicMatsubaraEnergyIndex
		= lowerFermionicMatsubaraEnergyIndex;
	this->upperFermionicMatsubaraEnergyIndex
		= upperFermionicMatsubaraEnergyIndex;
	this->lowerBosonicMatsubaraEnergyIndex
		= lowerBosonicMatsubaraEnergyIndex;
	this->upperBosonicMatsubaraEnergyIndex
		= upperBosonicMatsubaraEnergyIndex;
}

/*void BPropertyExtractor::saveEigenValues(string path, string filename){
	stringstream ss;
	ss << path;
	if(path.back() != '/')
		ss << '/';
	ss << filename;
	ofstream fout;
	fout.open(ss.str().c_str());
	for(int n = 0; n < dSolver->getModel()->getBasisSize(); n++){
		fout << dSolver->getEigenValues()[n] << "\n";
	}
	fout.close();
}

void DPropertyExtractor::getTabulatedHoppingAmplitudeSet(
	complex<double> **amplitudes,
	int **indices,
	int *numHoppingAmplitudes,
	int *maxIndexSize
){
	dSolver->getModel()->getHoppingAmplitudeSet()->tabulate(
		amplitudes,
		indices,
		numHoppingAmplitudes,
		maxIndexSize
	);
}*/

Property::EigenValues BlockDiagonalizer::getEigenValues(){
	int size = bSolver->getModel().getBasisSize();
//	const double *ev = bSolver->getEigenValues();

	Property::EigenValues eigenValues(size);
	std::vector<double> &data = eigenValues.getDataRW();
	for(int n = 0; n < size; n++)
		data[n] = bSolver->getEigenValue(n);
//		data[n] = ev[n];

	return eigenValues;
}

Property::WaveFunctions BlockDiagonalizer::calculateWaveFunctions(
	initializer_list<Index> patterns,
	initializer_list<int> states
){
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
		if(*states.begin() == IDX_ALL){
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
	for(unsigned int n = 0; n < patterns.size(); n++){
		TBTKAssert(
			(patterns.begin() + n)->split().size() == 2,
			"PropertyExtractor::BlockDiagonalizer::calculateGreensFunction()",
			"'pattern' must be 2 component Indices, but '"
			<< (patterns.begin() + n)->toString() << "' has "
			<< (patterns.begin() + n)->split().size() << "'"
			<< " component(s).",
			""
		);
	}

/*	IndexTree allIndices;
	IndexTree memoryLayout;
	for(unsigned int n = 0; n < patterns.size(); n++){
		const Index &pattern = *(patterns.begin() + n);

		std::vector<Index> components = pattern.split();
		TBTKAssert(
			components.size() == 2,
			"PropertyExtractor::BlockDiagonalizer::calculateGreensFunction()",
			"Invalid pattern '" << pattern.toString() << "'.",
			"The Index must be a compund Index with 2 component"
			<< " Indices, but the number of components are '"
			<< components.size() << "'."
		);

		//TODO
		//Continue here.
		const Index &toPattern = components[0];
		const Index &fromPattern = components[1];

		//Generate allIndices.
		IndexTree allToIndices = generateIndexTree(
			{toPattern},
			bSolver->getModel().getHoppingAmplitudeSet(),
			false,
			false
		);
		IndexTree allFromIndices = generateIndexTree(
			{fromPattern},
			bSolver->getModel().getHoppingAmplitudeSet(),
			false,
			false
		);
		for(
			IndexTree::ConstIterator iteratorTo
				= allToIndices.cbegin();
			iteratorTo != allToIndices.cend();
			++iteratorTo
		){
			for(
				IndexTree::ConstIterator iteratorFrom
					= allFromIndices.cbegin();
				iteratorFrom != allFromIndices.cend();
				++iteratorFrom
			){
				allIndices.add({*iteratorTo, *iteratorFrom});
			}
		}

		//Generate memory layout
		IndexTree memoryLayoutTo = generateIndexTree(
			{toPattern},
			bSolver->getModel().getHoppingAmplitudeSet(),
			true,
			false
		);
		IndexTree memoryLayoutFrom = generateIndexTree(
			{fromPattern},
			bSolver->getModel().getHoppingAmplitudeSet(),
			true,
			false
		);
		for(
			IndexTree::ConstIterator iteratorTo
				= memoryLayoutTo.cbegin();
			iteratorTo != memoryLayoutTo.cend();
			++iteratorTo
		){
			for(
				IndexTree::ConstIterator iteratorFrom
					= memoryLayoutFrom.cbegin();
				iteratorFrom != memoryLayoutFrom.cend();
				++iteratorFrom
			){
				memoryLayout.add({*iteratorTo, *iteratorFrom});
			}
		}
	}
	allIndices.generateLinearMap();
	memoryLayout.generateLinearMap();*/

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

		hint = &greensFunction;

		calculate(
			calculateGreensFunctionCallback,
			allIndices,
			memoryLayout,
			greensFunction
		);

		hint = nullptr;

		return greensFunction;
	}
	case Property::GreensFunction::Type::Matsubara:
	{
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
/*		unsigned int numMatsubaraEnergies = (
			upperFermionicMatsubaraEnergyIndex
			- lowerFermionicMatsubaraEnergyIndex
		)/2 + 1;*/

		double temperature = bSolver->getModel().getTemperature();
		double kT = UnitHandler::getK_BN()*temperature;
		double fundamentalMatsubaraEnergy = M_PI*kT;

/*		hint = new vector<complex<double>>();
		vector<complex<double>> &energies = *((vector<complex<double>>*)hint);
		energies.clear();
		energies.reserve(numMatsubaraEnergies);
		for(int n = 0; n < (int)numMatsubaraEnergies; n++){
			energies.push_back(
				(double)(
					lowerFermionicMatsubaraEnergyIndex
					+ 2*n
				)*complex<double>(0, 1)*M_PI*kT
			);
		}*/

		Property::GreensFunction greensFunction(
			memoryLayout,
			lowerFermionicMatsubaraEnergyIndex,
			upperFermionicMatsubaraEnergyIndex,
			fundamentalMatsubaraEnergy
		);

		hint = &greensFunction;

		calculate(
			calculateGreensFunctionCallback,
			allIndices,
			memoryLayout,
			greensFunction
		);

		hint = nullptr;

//		delete (vector<complex<double>>*)hint;

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
		energyType == EnergyType::Real,
		"PropertyExtractor::BlockDiagonalizer::calculateDOS()",
		"Only real energies supported for the DOS.",
		"Use PropertyExtractor::BlockDiagonalizer::setEnergyWindow()"
		<< " to set a real energy window."
	);
//	const double *ev = bSolver->getEigenValues();

	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	Property::DOS dos(lowerBound, upperBound, energyResolution);
	std::vector<double> &data = dos.getDataRW();
	double dE = (upperBound - lowerBound)/energyResolution;
	for(int n = 0; n < bSolver->getModel().getBasisSize(); n++){
		int e = (int)(((bSolver->getEigenValue(n) - lowerBound)/(upperBound - lowerBound))*energyResolution);
//		int e = (int)(((ev[n] - lowerBound)/(upperBound - lowerBound))*energyResolution);
		if(e >= 0 && e < energyResolution){
			data[e] += 1./dE;
		}
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

/*Property::Density DPropertyExtractor::calculateDensity(
	Index pattern,
	Index ranges
){
	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Density density(lDimensions, lRanges);

	calculate(calculateDensityCallback, (void*)density.getDataRW(), pattern, ranges, 0, 1);

	return density;
}*/

Property::Density BlockDiagonalizer::calculateDensity(
	initializer_list<Index> patterns
){
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

	calculate(
		calculateDensityCallback,
		allIndices,
		memoryLayout,
		density
	);

	return density;
}

/*Property::Magnetization DPropertyExtractor::calculateMagnetization(
	Index pattern,
	Index ranges
){
	hint = new int[1];
	((int*)hint)[0] = -1;
	for(unsigned int n = 0; n < pattern.size(); n++){
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
			"DPropertyExtractor::calculateMagnetization()",
			"No spin index indiceated.",
			"Used IDX_SPIN to indicate the position of the spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Magnetization magnetization(lDimensions, lRanges);

	calculate(calculateMAGCallback, (void*)magnetization.getDataRW(), pattern, ranges, 0, 4);

	delete [] (int*)hint;

	return magnetization;
}*/

Property::Magnetization BlockDiagonalizer::calculateMagnetization(
	std::initializer_list<Index> patterns
){
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

/*Property::LDOS BPropertyExtractor::calculateLDOS(
	Index pattern,
	Index ranges
){
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

	calculate(calculateLDOSCallback, (void*)ldos.getDataRW(), pattern, ranges, 0, energyResolution);

	return ldos;
}*/

Property::LDOS BlockDiagonalizer::calculateLDOS(
	std::initializer_list<Index> patterns
){
	TBTKAssert(
		energyType == EnergyType::Real,
		"PropertyExtractor::BlockDiagonalizer::calculateLDOS()",
		"Only real energies supported for the LDOS.",
		"Use PropertyExtractor::BlockDiagonalizer::setEnergyWindow()"
		<< " to set a real energy window."
	);

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

	calculate(
		calculateLDOSCallback,
		allIndices,
		memoryLayout,
		ldos
	);

	return ldos;
}

/*Property::SpinPolarizedLDOS DPropertyExtractor::calculateSpinPolarizedLDOS(
	Index pattern,
	Index ranges
){
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
	for(unsigned int n = 0; n < pattern.size(); n++){
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
			"DPropertyExtractor::calculateSpinPolarizedLDOS()",
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
		(void*)spinPolarizedLDOS.getDataRW(),
		pattern,
		ranges,
		0,
		4*energyResolution
	);

	delete [] ((double**)hint)[0];
	delete [] ((int**)hint)[1];
	delete [] (void**)hint;

	return spinPolarizedLDOS;
}*/

Property::SpinPolarizedLDOS BlockDiagonalizer::calculateSpinPolarizedLDOS(
	std::initializer_list<Index> patterns
){
	TBTKAssert(
		energyType == EnergyType::Real,
		"PropertyExtractor::BlockDiagonalizer::calculateSpinPolarizedLDOS()",
		"Only real energies supported for the SpinPolarizedLDOS.",
		"Use PropertyExtractor::BlockDiagonalizer::setEnergyWindow()"
		<< " to set a real energy window."
	);

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
	void *waveFunctions,
	const Index &index,
	int offset
){
	BlockDiagonalizer *pe = (BlockDiagonalizer*)cb_this;

	const vector<unsigned int> states = ((Property::WaveFunctions**)pe->hint)[0]->getStates();
	for(unsigned int n = 0; n < states.size(); n++)
		((complex<double>*)waveFunctions)[offset + n] += pe->getAmplitude(states.at(n), index);
}

void BlockDiagonalizer::calculateGreensFunctionCallback(
	PropertyExtractor *cb_this,
	void *greensFunction,
	const Index &index,
	int offset
){
	BlockDiagonalizer *propertyExtractor = (BlockDiagonalizer*)cb_this;
/*	const vector<complex<double>> &energies
		= *((vector<complex<double>>*)propertyExtractor->hint);*/

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

	Property::GreensFunction &gf
		= *(Property::GreensFunction*)propertyExtractor->hint;

	switch(gf.getType()){
	case Property::GreensFunction::Type::Advanced:
	case Property::GreensFunction::Type::Retarded:
	{
		double lowerBound = propertyExtractor->getLowerBound();
		double upperBound = propertyExtractor->getUpperBound();
		double energyResolution
			= propertyExtractor->getEnergyResolution();
		double dE = (upperBound - lowerBound)/energyResolution;
		double delta = propertyExtractor->getEnergyInfinitesimal();
		if(gf.getType() == Property::GreensFunction::Type::Advanced)
			delta *= -1;

		for(int e = 0; e < energyResolution; e++){
			double E = lowerBound +e*dE;
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
				((complex<double>*)greensFunction)[offset + e]
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
			= gf.getNumMatsubaraEnergies();

		for(unsigned int n = firstStateInBlock; n <= lastStateInBlock; n++){
			double energy = propertyExtractor->bSolver->getEigenValue(n);
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
				((complex<double>*)greensFunction)[offset + e]
					+= numerator/(
						gf.getMatsubaraEnergy(e)
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
	void* density,
	const Index &index,
	int offset
){
	BlockDiagonalizer *pe = (BlockDiagonalizer*)cb_this;

//	const double *eigen_values = pe->bSolver->getEigenValues();
	Statistics statistics = pe->bSolver->getModel().getStatistics();
	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
//	for(int n = 0; n < pe->bSolver->getModel()->getBasisSize(); n++){
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

		((double*)density)[offset] += pow(abs(u), 2)*weight;
	}
}

void BlockDiagonalizer::calculateMagnetizationCallback(
	PropertyExtractor *cb_this,
	void *mag,
	const Index &index,
	int offset
){
	BlockDiagonalizer *pe = (BlockDiagonalizer*)cb_this;

//	const double *eigen_values = pe->bSolver->getEigenValues();
	Statistics statistics = pe->bSolver->getModel().getStatistics();

	int spin_index = ((int*)pe->hint)[0];
	Index index_u(index);
	Index index_d(index);
	index_u.at(spin_index) = 0;
	index_d.at(spin_index) = 1;
	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
//	for(int n = 0; n < pe->bSolver->getModel()->getBasisSize(); n++){
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

/*		((complex<double>*)mag)[offset + 0] += conj(u_u)*u_u*weight;
		((complex<double>*)mag)[offset + 1] += conj(u_u)*u_d*weight;
		((complex<double>*)mag)[offset + 2] += conj(u_d)*u_u*weight;
		((complex<double>*)mag)[offset + 3] += conj(u_d)*u_d*weight;*/
		((SpinMatrix*)mag)[offset].at(0, 0) += conj(u_u)*u_u*weight;
		((SpinMatrix*)mag)[offset].at(0, 1) += conj(u_u)*u_d*weight;
		((SpinMatrix*)mag)[offset].at(1, 0) += conj(u_d)*u_u*weight;
		((SpinMatrix*)mag)[offset].at(1, 1) += conj(u_d)*u_d*weight;
	}
}

void BlockDiagonalizer::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	void *ldos,
	const Index &index,
	int offset
){
	BlockDiagonalizer *pe = (BlockDiagonalizer*)cb_this;

//	const double *eigen_values = pe->bSolver->getEigenValues();

	double u_lim = ((double**)pe->hint)[0][0];
	double l_lim = ((double**)pe->hint)[0][1];
	int resolution = ((int**)pe->hint)[1][0];

	double step_size = (u_lim - l_lim)/(double)resolution;

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();

	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	double dE = (upperBound - lowerBound)/energyResolution;
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
//	for(int n = 0; n < pe->bSolver->getModel()->getBasisSize(); n++){
		double eigenValue = pe->bSolver->getEigenValue(n);
		if(eigenValue > l_lim && eigenValue < u_lim){
			complex<double> u = pe->bSolver->getAmplitude(n, index);

			int e = (int)((eigenValue - l_lim)/step_size);
			if(e >= resolution)
				e = resolution-1;
			((double*)ldos)[offset + e] += real(conj(u)*u)/dE;
		}
	}
}

void BlockDiagonalizer::calculateSP_LDOSCallback(
	PropertyExtractor *cb_this,
	void *sp_ldos,
	const Index &index,
	int offset
){
	BlockDiagonalizer *pe = (BlockDiagonalizer*)cb_this;

	double u_lim = ((double**)pe->hint)[0][0];
	double l_lim = ((double**)pe->hint)[0][1];
	int resolution = ((int**)pe->hint)[1][0];
	int spin_index = ((int**)pe->hint)[1][1];

	double step_size = (u_lim - l_lim)/(double)resolution;

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();

	Index index_u(index);
	Index index_d(index);
	index_u.at(spin_index) = 0;
	index_d.at(spin_index) = 1;
	int firstStateInBlock = pe->bSolver->getFirstStateInBlock(index);
	int lastStateInBlock = pe->bSolver->getLastStateInBlock(index);
	double dE = (upperBound - lowerBound)/energyResolution;
	for(int n = firstStateInBlock; n <= lastStateInBlock; n++){
//	for(int n = 0; n < pe->bSolver->getModel()->getBasisSize(); n++){
		double eigenValue = pe->bSolver->getEigenValue(n);
		if(eigenValue > l_lim && eigenValue < u_lim){
			complex<double> u_u = pe->bSolver->getAmplitude(n, index_u);
			complex<double> u_d = pe->bSolver->getAmplitude(n, index_d);

			int e = (int)((eigenValue - l_lim)/step_size);
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
